from utils.dataset.normalization import normalize_series
from utils.modeling import bert_clf
import pandas as pd
import json
import time
import numpy as np
import torch
import os
import gc


def extract_model_info(model_path):
    base = os.path.basename(model_path)
    model_name = os.path.splitext(base)[0]
    result = dict()
    result['model_name'] = model_name
    """
    0: pretrained bert name
    1: keep emojis?
    2: segment hashtag?
    3: batch size
    4: learning rate
    5: epoch
    6: random state
    7: precision score
    8: recall score
    """
    config = model_name.split('_')
    assert len(config) == 9
    result['pretrained_bert'] = "/".join(config[0].split('+'))
    result['keep_emojis'] = bool(int(config[1]))
    result['segment_hashtag'] = bool(int(config[2]))
    result['batch_size'] = int(config[3])
    result['learning_rate'] = float(config[4])
    result['epoch'] = int(config[5])
    result['random_state'] = int(config[6])
    result['precision'] = float(config[7]) / 10000
    result['recall'] = float(config[8]) / 10000
    return result


def preprocess(df_path, pretrained_bert_name,
               labels=None,
               to_lower=None, to_ascii=None,
               keep_emojis=None, segment_hashtag=None,
               username=None, httpurl=None):
    if labels is None:
        labels = [0, 1]
    label_map = dict()
    label_map[labels[0]] = 0
    label_map[labels[1]] = 1
    additional_config_dirpath = f'./config/bert-reconfig/{pretrained_bert_name}'
    with open(os.path.join(additional_config_dirpath, 'preprocessing.json')) as JSON:
        preprocess_config = json.loads(JSON.read())
    if to_lower is None:
        to_lower = preprocess_config['to_lower']
    if to_ascii is None:
        to_ascii = preprocess_config['to_ascii']
    if keep_emojis is None:
        keep_emojis = preprocess_config['keep_emojis']
    if username is None:
        username = preprocess_config['username']
    if httpurl is None:
        httpurl = preprocess_config['httpurl']
    if segment_hashtag is None:
        segment_hashtag = preprocess_config['segment_hashtag']
    df = pd.read_csv(df_path, sep='\t', header=None).drop(columns=[0])
    df[1] = normalize_series(df[1],
                             to_lower=to_lower,
                             to_ascii=to_ascii,
                             keep_emojis=keep_emojis,
                             segment_hashtag=segment_hashtag,
                             username=username,
                             httpurl=httpurl)
    if 2 in df.columns:
        df[2] = df[2].apply(lambda label: label_map[label])
    else:
        df[2] = -1
    temp_path = os.path.abspath(f'./temp{str(int(time.time()))}.tsv')
    df.to_csv(temp_path, sep='\t', header=False, index=False)
    return temp_path


def single_predict(model_path, df_path, labels=None):
    if labels is None:
        labels = [0, 1]
    model_info = extract_model_info(model_path)
    print('-' * 20)
    print('Processing', model_info['model_name'])
    del model_info['model_name']
    print('Model info:', model_info)
    print('-- Preparing data for inference...')
    preprocessed_df_path = preprocess(df_path=df_path,
                                      labels=labels,
                                      pretrained_bert_name=model_info['pretrained_bert'],
                                      keep_emojis=model_info['keep_emojis'],
                                      segment_hashtag=model_info['segment_hashtag'])
    print('-- Predicting...')
    _, predictions, predictions_proba, _ = bert_clf.predict(
        pretrained_bert_name=model_info['pretrained_bert'],
        model_path=model_path,
        batch_size=model_info['batch_size'],
        random_state=model_info['random_state'],
        df_path=preprocessed_df_path)
    os.remove(preprocessed_df_path)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions


def soft_voting_predict(all_model_paths, df_path, labels=None):
    if labels is None:
        labels = [0, 1]
    all_model_predictions = []
    all_model_probas = []
    all_model_precision_scores = []
    all_models_recall_scores = []
    for model_path in all_model_paths:
        model_info = extract_model_info(model_path)
        print('-' * 20)
        print('Processing', model_info['model_name'])
        del model_info['model_name']
        print('Model info:', model_info)
        all_model_precision_scores.append(model_info['precision'])
        all_models_recall_scores.append(model_info['recall'])
        print('-- Preparing data for inference...')
        preprocessed_df_path = preprocess(df_path=df_path,
                                          labels=labels,
                                          pretrained_bert_name=model_info['pretrained_bert'],
                                          keep_emojis=model_info['keep_emojis'],
                                          segment_hashtag=model_info['segment_hashtag'])
        print('-- Predicting...')
        _, predictions, predictions_proba, _ = bert_clf.predict(
            pretrained_bert_name=model_info['pretrained_bert'],
            model_path=model_path,
            batch_size=model_info['batch_size'],
            random_state=model_info['random_state'],
            df_path=preprocessed_df_path)
        all_model_predictions.append(predictions)
        all_model_probas.append(predictions_proba)
        os.remove(preprocessed_df_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    shape = all_model_probas[0].shape
    for proba in all_model_probas:
        assert proba.shape == shape

    proba_final = np.zeros(shape=shape)
    for prediction_idx in range(proba_final.shape[0]):
        for model_idx in range(len(all_model_probas)):
            proba_final[prediction_idx][0] += all_model_probas[model_idx][prediction_idx][0]
            proba_final[prediction_idx][1] += all_model_probas[model_idx][prediction_idx][1]
    predictions_final = proba_final.argmax(axis=1)
    predictions_final = np.vectorize(lambda label: labels[label])(predictions_final)
    gc.collect()
    return predictions_final


def hard_voting_predict(all_model_paths, df_path, labels=None):
    if labels is None:
        labels = [0, 1]
    all_model_predictions = []
    for model_path in all_model_paths:
        model_info = extract_model_info(model_path)
        print('-' * 20)
        print('Processing', model_info['model_name'])
        del model_info['model_name']
        print('Model info:', model_info)
        print('-- Preparing data for inference...')
        preprocessed_df_path = preprocess(df_path=df_path,
                                          labels=labels,
                                          pretrained_bert_name=model_info['pretrained_bert'],
                                          keep_emojis=model_info['keep_emojis'],
                                          segment_hashtag=model_info['segment_hashtag'])
        print('-- Predicting...')
        _, predictions, _, _ = bert_clf.predict(
            pretrained_bert_name=model_info['pretrained_bert'],
            model_path=model_path,
            batch_size=model_info['batch_size'],
            random_state=model_info['random_state'],
            df_path=preprocessed_df_path)
        all_model_predictions.append(predictions)
        os.remove(preprocessed_df_path)
        gc.collect()

    predictions_final = np.zeros_like(all_model_predictions[0])
    for prediction_idx in range(predictions_final.shape[0]):
        votes = [0, 0]
        for model_idx in range(len(all_model_predictions)):
            votes[all_model_predictions[model_idx][prediction_idx]] += 1
        predictions_final[prediction_idx] = np.argmax(votes)
    predictions_final = np.vectorize(lambda label: labels[label])(predictions_final)
    gc.collect()
    return predictions_final

