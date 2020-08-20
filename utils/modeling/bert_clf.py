import os
import gc
import json
import random
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score, accuracy_score
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
        b_tp, b_fp, b_tn, b_fn = perf_measure(
            targets.detach().cpu().numpy(),
            preds.detach().cpu().numpy()
        )
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

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0
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
            b_tp, b_fp, b_tn, b_fn = perf_measure(
                targets.detach().cpu().numpy(),
                preds.detach().cpu().numpy()
            )
            tn += b_tn
            fp += b_fp
            fn += b_fn
            tp += b_tp

            losses.append(loss.item())

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0
    return f1, precision, recall, accuracy, np.mean(losses)


def predict_loaded_model(model, tokenizer, encode_config,
                         df_path=os.path.join(__dataset_path__, 'normalized/test_normalized.tsv'),
                         batch_size=16, random_state=42, device=device):
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)

    df = pd.read_csv(df_path,
                     sep='\t',
                     header=None)  # check
    dataset = BertDataset(texts=df[0].values,
                          labels=df[1].values,
                          tokenizer=tokenizer,
                          encode_config=encode_config)
    data_loader = create_data_loader(dataset, batch_size)

    model = model.to(device)
    model = model.eval()

    texts = []
    predictions = []
    prediction_probs = []
    real_labels = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
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
            real_labels.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_labels = torch.stack(real_labels).cpu()
    return texts, predictions, prediction_probs, real_labels


def predict(pretrained_bert_name, model_path,
            batch_size=16, random_state=42, device=device,
            df_path=os.path.join(__dataset_path__, 'normalized/test_normalized.tsv')):
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    model = load_sequence_classification_model(pretrained_bert_name)
    tokenizer, encode_config = load_pretrained_tokenization(pretrained_bert_name)

    model.load_state_dict(torch.load(model_path, map_location=device))
    return predict_loaded_model(model=model, tokenizer=tokenizer, encode_config=encode_config,
                                df_path=df_path,
                                batch_size=batch_size, random_state=random_state, device=device)


def eval(pretrained_bert, model_path, batch_size=16, random_state=42, device=device,
         df_path=os.path.join(__dataset_path__, 'normalized/test_normalized.tsv')):
    y_review_texts, y_pred, y_pred_probs, y_test = predict(
        pretrained_bert_name=pretrained_bert,
        batch_size=batch_size,
        model_path=model_path,
        random_state=random_state,
        device=device,
        df_path=df_path
    )
    print('F1       :', f1_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred))
    print('Recall   :', recall_score(y_test, y_pred))
    print('Accuracy :', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def perf_measure(y_true, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_true[i] != y_pred[i]:
            FP += 1
        if y_true[i] == y_pred[i] == 0:
            TN += 1
        if y_pred[i] == 0 and y_true[i] != y_pred[i]:
            FN += 1
    return TP, FP, TN, FN


def train_util(pretrained_bert_name, train_data_loader, valid_data_loader, mode="training", kth_fold=None,
               batch_size=16, learning_rate=2e-5, epochs=4, random_state=42, device=device):
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)

    assert mode in ["training", "tuning"]
    save_path_prefix = mode  # tuning
    save_path_prefix = os.path.join(pretrained_bert_name, save_path_prefix)  # vinai/bertweet-base/tuning/
    save_path_prefix = os.path.join(__models_path__, save_path_prefix)  # models/vinai/bertweet-base/tuning
    # models/vinai/bertweet-base/tuning/16_2e-5_4_42
    save_path_prefix = os.path.join(save_path_prefix, f"{batch_size}_{learning_rate}_{epochs}_{random_state}/")
    if kth_fold is None:
        # models/vinai/bertweet-base/tuning/16_2e-5_4_42/non_kfold
        save_path_prefix = os.path.join(save_path_prefix, "non_kfold")
    else:
        # models/vinai/bertweet-base/tuning/16_2e-5_4_42/0th_fold
        save_path_prefix = os.path.join(save_path_prefix, f"{kth_fold}th_fold")
    save_path_prefix = os.path.abspath(save_path_prefix)

    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    model = load_sequence_classification_model(pretrained_bert_name)
    model = model.to(device)

    weight_decay = 0.00001
    optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False, weight_decay=weight_decay)
    total_steps = len(train_data_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps / 10,
        num_training_steps=total_steps
    )

    history = defaultdict(list)
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 20)

        train_f1, train_precision, train_recall, train_accuracy, train_loss = train_epoch(
            model=model,
            data_loader=train_data_loader,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler
        )
        train_report = {
            'loss': train_loss,
            'precision': train_precision,
            'recall': train_recall,
            'f1': train_f1,
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
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1,
            'accuracy': val_accuracy
        }
        print('Valid:', val_report)

        history['train_reports'].append(train_report)
        history['val_reports'].append(val_report)

        if mode == 'tuning':
            path_to_save = os.path.join(save_path_prefix, f"model_epoch_{epoch + 1}.bin")
            torch.save(model.state_dict(), path_to_save)
            history["model_paths"].append(path_to_save)
            print(f'State of model after epoch {epoch + 1} saved at', path_to_save)
        print()

    model_cpu = model.to(torch.device('cpu'))
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if mode == "training":
        path_to_save = os.path.abspath(os.path.join(save_path_prefix, "model.bin"))
        torch.save(model_cpu.state_dict(), path_to_save)
        print(f'State of trained model saved at', path_to_save)
        return model_cpu, history
    return history


def train(pretrained_bert_name, batch_size=16, learning_rate=2e-5, epochs=10, random_state=42, device=device,
          train_ds_path=os.path.join(__dataset_path__, 'normalized/train_normalized.tsv'),
          valid_ds_path=os.path.join(__dataset_path__, 'normalized/valid_normalized.tsv'),
          kfold_num=0, kth_specified=None):
    tokenizer, encode_config = load_pretrained_tokenization(pretrained_bert_name)
    train_df = pd.read_csv(train_ds_path,
                           sep='\t',
                           header=None)  # check
    valid_df = pd.read_csv(valid_ds_path,
                           sep='\t',
                           header=None)  # check
    print('Using pretrained bert model:', pretrained_bert_name)
    params = {
        'pretrained_bert': pretrained_bert_name,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'random_state': random_state,
        'max_seq': encode_config['max_length']
    }
    print('Params =', params)
    histories = []
    if kfold_num != 0:
        kf = StratifiedKFold(n_splits=kfold_num, random_state=random_state, shuffle=True)
        data = pd.concat([train_df, valid_df]).reset_index(drop=True)
        for kth_fold, split_indices in enumerate(kf.split(data, y=data[1].values)):
            train_indices, val_indices = split_indices
            train_df = data.iloc[train_indices]
            valid_df = data.iloc[val_indices]
            print(f"{kth_fold}th "
                  "KFOLD----------------------------------------------------------------------------------------------")
            print("Train dataset's info:")
            # print(train_df.info())
            print(train_df[1].value_counts())
            print()
            print("Valid dataset's info:")
            # print(valid_df.info())
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
            history = train_util(pretrained_bert_name=pretrained_bert_name,
                                 train_data_loader=train_data_loader,
                                 valid_data_loader=valid_data_loader,
                                 batch_size=batch_size,
                                 learning_rate=learning_rate,
                                 epochs=epochs,
                                 random_state=random_state,
                                 device=device,
                                 mode="tuning",
                                 kth_fold=kth_fold)
            history['params'] = params
            history['params']['kth_fold'] = kth_fold
            histories.append(history)
            print()
            print()
        now = datetime.now()
        dt_string = now.strftime("%H:%M-%d-%m")
        with open(os.path.join(
                __models_path__,
                f'{pretrained_bert_name}/history_{dt_string}.json'),
                'w') as fout:
            json.dump(histories, fout)
        return histories

    else:
        if kth_specified is not None:
            kf = StratifiedKFold(n_splits=kfold_num, random_state=random_state, shuffle=True)
            data = pd.concat([train_df, valid_df]).reset_index(drop=True)
            train_indices, val_indices = list(kf.split(data, y=data[1].values))[kth_specified]
            train_df = data.iloc[train_indices]
            valid_df = data.iloc[val_indices]
            print(f"Split dataset by kfold, using {kth_specified} fold, random state = {random_state}")
        print("Train dataset's info:")
        # print(train_df.info())
        print(train_df[1].value_counts())
        print()
        print("Valid dataset's info:")
        # print(valid_df.info())
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
        model, history = train_util(pretrained_bert_name=pretrained_bert_name,
                                    train_data_loader=train_data_loader,
                                    valid_data_loader=valid_data_loader,
                                    batch_size=batch_size,
                                    learning_rate=learning_rate,
                                    epochs=epochs,
                                    random_state=random_state,
                                    device=device,
                                    kth_fold=kth_specified)
        return model, history
