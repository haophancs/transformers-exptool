import os
import gc
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score, accuracy_score, \
    confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification

from utils import __models_path__, __dataset_path__
from utils.dataset import create_data_loader, BertDataset
from utils.pretrained import load_pretrained_tokenization, load_pretrained_model, load_pretrain_model_config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_sequence_classification_model(pretrained_name):
    if not os.path.exists(os.path.join(__models_path__, f'pretrained/{pretrained_name}/pytorch_model.bin')):
        loaded_pretrained_model = load_pretrained_model(pretrained_name)
        loaded_pretrained_model.save_pretrained(
            os.path.join(__models_path__, f'pretrained/{pretrained_name}')
        )
    model_config = load_pretrain_model_config(pretrained_name)
    model_clf = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(__models_path__, f'pretrained/{pretrained_name}/pytorch_model.bin'),
        config=model_config
    )
    return model_clf


def train_epoch(
        model,
        data_loader,
        optimizer,
        device,
        scheduler):
    model = model.train()

    losses = []
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["label"].to(device)

        loss, logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=targets
        )

        _, preds = torch.max(logits, dim=1)
        b_tn, b_fp, b_fn, b_tp = confusion_matrix(
            targets.detach().cpu().numpy(),
            preds.detach().cpu().numpy()
        ).ravel()
        tn += b_tn
        fp += b_fp
        fn += b_fn
        tp += b_tp

        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return f1, precision, recall, accuracy, np.mean(losses)


def eval_epoch(model, data_loader, device):
    model = model.eval()

    losses = []
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["label"].to(device)

            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=targets
            )
            _, preds = torch.max(logits, dim=1)
            b_tn, b_fp, b_fn, b_tp = confusion_matrix(
                targets.detach().cpu().numpy(),
                preds.detach().cpu().numpy()
            ).ravel()
            tn += b_tn
            fp += b_fp
            fn += b_fn
            tp += b_tp

            losses.append(loss.item())

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return f1, precision, recall, accuracy, np.mean(losses)


def predict(pretrained_bert_name, batch_size=16, learning_rate=2e-5, epochs=10, random_state=42, device=device,
            test_ds_path=os.path.join(__dataset_path__, 'normalized/test_normalized.tsv')):
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    test_df = pd.read_csv(test_ds_path,
                          sep='\t',
                          header=None)  # check
    tokenizer, encode_config = load_pretrained_tokenization(pretrained_bert_name)

    with_labels = 1 in test_df.columns
    if with_labels:
        test_dataset = BertDataset(texts=test_df[0].values,
                                   labels=test_df[1].values,
                                   tokenizer=tokenizer,
                                   encode_config=encode_config)
    else:
        test_dataset = BertDataset(texts=test_df[0].values,
                                   tokenizer=tokenizer,
                                   encode_config=encode_config)
    test_data_loader = create_data_loader(test_dataset, batch_size)
    model = load_sequence_classification_model(pretrained_bert_name)
    model.load_state_dict(
        torch.load(
            os.path.join(__models_path__,
                         f'./{pretrained_bert_name}/{batch_size}_{learning_rate}_{epochs}_{random_state}.bin'),
            map_location=device)
    )
    model = model.to(device)
    model = model.eval()

    texts = []
    predictions = []
    prediction_probs = []
    real_labels = []

    with torch.no_grad():
        for d in test_data_loader:
            texts = d["text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            if with_labels:
                targets = d["label"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )[0]
            _, preds = torch.max(logits, dim=1)

            probs = F.softmax(logits, dim=1)

            texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            if with_labels:
                real_labels.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    if with_labels:
        real_labels = torch.stack(real_labels).cpu()
        return texts, predictions, prediction_probs, real_labels
    return texts, predictions, prediction_probs


def eval(pretrained_bert, batch_size=16, learning_rate=2e-5, epochs=10, random_state=42, device=device,
         test_ds_path=os.path.join(__dataset_path__, 'normalized/test_normalized.tsv')):
    y_review_texts, y_pred, y_pred_probs, y_test = predict(
        pretrained_bert_name=pretrained_bert,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        random_state=random_state,
        device=device,
        test_ds_path=test_ds_path
    )
    print('F1       :', f1_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred))
    print('Recall   :', recall_score(y_test, y_pred))
    print('Accuracy :', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


def _train(pretrained_bert_name, train_data_loader, valid_data_loader,
           batch_size=16, learning_rate=2e-5, epochs=4, random_state=42, device=device):
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)

    model = load_sequence_classification_model(pretrained_bert_name)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00001)
    total_steps = len(train_data_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=2 * len(train_data_loader),
        num_training_steps=total_steps
    )

    best_f1_score = -1
    history = defaultdict(list)

    for epoch in range(epochs):

        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        train_f1, train_precision, train_recall, train_accuracy, train_loss = train_epoch(
            model=model,
            data_loader=train_data_loader,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler
        )
        train_report = {
            'loss': train_loss,
            'f1': train_f1,
            'precision': train_precision,
            'recall': train_recall,
            'accuracy': train_accuracy
        }
        print('Train:', train_report)

        val_f1, val_precision, val_recall, val_accuracy, val_loss = eval_epoch(
            model=model,
            data_loader=valid_data_loader,
            device=device
        )

        val_report = {
            'loss': val_loss,
            'f1': val_f1,
            'precision': val_precision,
            'recall': val_recall,
            'accuracy': val_accuracy
        }
        print('Train:', val_report)
        print()

        history['train_f1'].append(train_f1)
        history['train_precision'].append(train_precision)
        history['train_recall'].append(train_recall)
        history['train_accuracy'].append(train_accuracy)
        history['train_loss'].append(train_loss)

        history['val_f1'].append(val_f1)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_accuracy'].append(val_accuracy)
        history['val_loss'].append(train_loss)

        if val_f1 > best_f1_score and epoch >= 1:
            if not os.path.exists(os.path.join(__models_path__, f'{pretrained_bert_name}')):
                os.makedirs(os.path.join(__models_path__, f'{pretrained_bert_name}/'))
            torch.save(
                model.state_dict(),
                os.path.join(__models_path__,
                             f'{pretrained_bert_name}/{batch_size}_{learning_rate}_{epochs}_{random_state}.bin'))
            best_f1_score = val_f1

    model_cpu = model.to(torch.device('cpu'))
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model_cpu, history


def train(pretrained_bert_name, batch_size=16, learning_rate=2e-5, epochs=10, random_state=42, device=device,
          train_ds_path=os.path.join(__dataset_path__, 'normalized/train_normalized.tsv'),
          valid_ds_path=os.path.join(__dataset_path__, 'normalized/valid_normalized.tsv'),
          kfold_num=0):
    tokenizer, encode_config = load_pretrained_tokenization(pretrained_bert_name)
    train_df = pd.read_csv(train_ds_path,
                           sep='\t',
                           header=None)  # check
    valid_df = pd.read_csv(valid_ds_path,
                           sep='\t',
                           header=None)  # check
    print('Using pretrained bert model:', pretrained_bert_name)
    print('Params = {',
          f'batch_size: {batch_size},',
          f'learning_rate: {learning_rate},',
          f'epochs: {epochs},',
          f'random_state: {random_state}',
          f"max sequence length: {encode_config['max_length']}",
          '}')
    if kfold_num != 0:
        kf = StratifiedKFold(n_splits=kfold_num, random_state=random_state, shuffle=True)
        data = pd.concat([train_df, valid_df]).reset_index(drop=True)
        histories = []
        models = []
        for train_indices, val_indices in kf.split(data, y=data[1].values):
            train_df = data.iloc[train_indices]
            valid_df = data.iloc[val_indices]
            print("--------------------------NEW KFOLD------------------------------")
            print("Train dataset's info:")
            print(train_df.info())
            print(train_df[1].value_counts())
            print()
            print("Valid dataset's info:")
            print(valid_df.info())
            print(valid_df[1].value_counts())
            print('-' * 10)
            train_dataset = BertDataset(texts=train_df[0].values,
                                        labels=train_df[1].values,
                                        tokenizer=tokenizer,
                                        encode_config=encode_config)
            valid_dataset = BertDataset(texts=valid_df[0].values,
                                        labels=valid_df[1].values,
                                        tokenizer=tokenizer,
                                        encode_config=encode_config)
            train_data_loader = create_data_loader(train_dataset, batch_size)
            valid_data_loader = create_data_loader(valid_dataset, batch_size)
            model, history = _train(pretrained_bert_name=pretrained_bert_name,
                                    train_data_loader=train_data_loader,
                                    valid_data_loader=valid_data_loader,
                                    batch_size=batch_size,
                                    learning_rate=learning_rate,
                                    epochs=epochs,
                                    random_state=random_state,
                                    device=device)
            models.append(model)
            histories.append(history)
            print()
        return models, histories
    else:
        print("Train dataset's info:")
        print(train_df.info())
        print(train_df[1].value_counts())
        print()
        print("Valid dataset's info:")
        print(valid_df.info())
        print(valid_df[1].value_counts())
        print('-' * 10)
        train_dataset = BertDataset(texts=train_df[0].values,
                                    labels=train_df[1].values,
                                    tokenizer=tokenizer,
                                    encode_config=encode_config)
        valid_dataset = BertDataset(texts=valid_df[0].values,
                                    labels=valid_df[1].values,
                                    tokenizer=tokenizer,
                                    encode_config=encode_config)
        train_data_loader = create_data_loader(train_dataset, batch_size)
        valid_data_loader = create_data_loader(valid_dataset, batch_size)
        model, history = _train(pretrained_bert_name=pretrained_bert_name,
                                train_data_loader=train_data_loader,
                                valid_data_loader=valid_data_loader,
                                batch_size=batch_size,
                                learning_rate=learning_rate,
                                epochs=epochs,
                                random_state=random_state,
                                device=device)
        return [model], [history]
