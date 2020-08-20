import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score, accuracy_score
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
    f1_scores = []
    precision_scores = []
    recall_scores = []
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
        f1_scores.append(f1_score(targets.detach().cpu().numpy(),
                                  preds.detach().cpu().numpy()))
        precision_scores.append(precision_score(targets.detach().cpu().numpy(),
                                                preds.detach().cpu().numpy()))
        recall_scores.append(recall_score(targets.detach().cpu().numpy(),
                                          preds.detach().cpu().numpy()))
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return np.mean(f1_scores), np.mean(precision_scores), np.mean(recall_scores), np.mean(losses)


def eval_epoch(model, data_loader, device):
    model = model.eval()

    losses = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
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
            f1_scores.append(f1_score(targets.detach().cpu().numpy(),
                                      preds.detach().cpu().numpy()))
            precision_scores.append(precision_score(targets.detach().cpu().numpy(),
                                                    preds.detach().cpu().numpy()))
            recall_scores.append(recall_score(targets.detach().cpu().numpy(),
                                      preds.detach().cpu().numpy()))
            losses.append(loss.item())
    return np.mean(f1_scores), np.mean(precision_scores), np.mean(recall_scores), np.mean(losses)


def predict(pretrained_bert_name, model_path, batch_size=16, random_state=42,
            df_path=os.path.join(__dataset_path__, 'normalized/valid_normalized.tsv'),
            device=device):
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    test_df = pd.read_csv(df_path,
                          sep='\t',
                          header=None)  # check
    tokenizer, encode_config = load_pretrained_tokenization(pretrained_bert_name)
    test_dataset = BertDataset(texts=test_df[0].values,
                               labels=test_df[1].values,
                               tokenizer=tokenizer,
                               encode_config=encode_config)
    test_data_loader = create_data_loader(test_dataset, batch_size)

    model = load_sequence_classification_model(pretrained_bert_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
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


def eval(pretrained_bert_name, batch_size=16, learning_rate=2e-5, epochs=10, random_state=42,
         df_path=os.path.join(__dataset_path__, 'normalized/valid_normalized.tsv'),
         device=device):
    model_path = os.path.join(__models_path__,
                              f"{pretrained_bert_name}/{batch_size}_{learning_rate}_{epochs}_{random_state}.bin")
    y_review_texts, y_pred, y_pred_probs, y_test = predict(
        pretrained_bert_name=pretrained_bert_name,
        model_path=model_path,
        batch_size=batch_size,
        random_state=random_state,
        df_path=df_path,
        device=device
    )
    print('F1       :', f1_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred))
    print('Recall   :', recall_score(y_test, y_pred))
    print('Accuracy :', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def train(pretrained_bert_name, batch_size=16, learning_rate=2e-5, epochs=10, random_state=42, device=device):
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)

    tokenizer, encode_config = load_pretrained_tokenization(pretrained_bert_name)
    train_df = pd.read_csv(os.path.join(__dataset_path__, 'normalized/train_normalized.tsv'),
                           sep='\t',
                           header=None)  # check
    valid_df = pd.read_csv(os.path.join(__dataset_path__, 'normalized/valid_normalized.tsv'),
                           sep='\t',
                           header=None)  # check
    print('Using pretrained bert model:', pretrained_bert_name)
    print('Params = {',
          f'batch_size: {batch_size},',
          f'learning_rate: {learning_rate},',
          f'epochs: {epochs},',
          f'random_state: {random_state}',
          '}')

    train_dataset = BertDataset(texts=train_df[0].values,
                                labels=train_df[1].values,
                                tokenizer=tokenizer,
                                encode_config=encode_config)
    valid_dataset = BertDataset(texts=valid_df[0].values,
                                labels=valid_df[1].values,
                                tokenizer=tokenizer,
                                encode_config=encode_config)
    train_data_loader = create_data_loader(train_dataset, batch_size)
    val_data_loader = create_data_loader(valid_dataset, batch_size)

    model = load_sequence_classification_model(pretrained_bert_name)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)
    total_steps = len(train_data_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=2 * len(train_data_loader),
        num_training_steps=total_steps
    )

    history = defaultdict(list)
    for epoch in range(epochs):

        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        train_f1, train_precision, train_recall, train_loss = train_epoch(
            model=model,
            data_loader=train_data_loader,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler
        )

        print(f'Train loss {train_loss} f1_score {train_f1}, precision_score {train_precision}, recall_score {train_recall}')

        val_f1, val_precision, val_recall, val_loss = eval_epoch(
            model=model,
            data_loader=val_data_loader,
            device=device
        )

        print(f'Val loss {train_loss} f1_score {train_f1}, precision_score {train_precision}, recall_score {train_recall}')
        print()

        history['train_f1'].append(train_f1)
        history['train_precision'].append(train_precision)
        history['train_recall'].append(train_recall)
        history['train_loss'].append(train_loss)
        history['val_f1'].append(val_f1)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_loss'].append(val_loss)

        if not os.path.exists(os.path.join(__models_path__, f'{pretrained_bert_name}')):
            os.makedirs(os.path.join(__models_path__, f'{pretrained_bert_name}/'))
        torch.save(
            model.state_dict(),
            os.path.join(__models_path__,
                         f'{pretrained_bert_name}/{batch_size}_{learning_rate}_{epochs}_{random_state}.bin'))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, history