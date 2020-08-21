from utils.dataset.normalization import *
from utils.modeling import bert_clf
import numpy as np
import pandas as pd
import argparse
import json
import time
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
               labels=[0, 1],
               to_lower=None, to_ascii=None,
               keep_emojis=None, segment_hashtag=None,
               username=None, httpurl=None):
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
    df[2] = df[2].apply(lambda label: label_map[label])
    temp_path = os.path.abspath(f'./temp{str(int(time.time()))}.tsv')
    df.to_csv(temp_path, sep='\t', header=False, index=False)
    return temp_path


def ensemble_predict(model_paths, df_path, labels=[0, 1]):
    models_predictions = []
    models_probas = []
    models_precision_scores = []
    models_recall_scores = []
    for model_path in model_paths:
        model_info = extract_model_info(model_path)
        print('-' * 20)
        print('Processing', model_info['model_name'])
        del model_info['model_name']
        print('Model info:', model_info)
        models_precision_scores.append(model_info['precision'])
        models_recall_scores.append(model_info['recall'])
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
        models_predictions.append(predictions)
        models_probas.append(predictions_proba)
        os.remove(preprocessed_df_path)
        gc.collect()

    shape = models_probas[0].shape
    for proba in models_probas:
        assert proba.shape == shape

    models_precision_scores = np.array(models_precision_scores)
    models_recall_scores = np.array(models_recall_scores)
    proba_final = np.zeros(shape=shape)
    for prediction_idx in range(proba_final.shape[0]):
        for model_idx in range(len(models_probas)):
            proba_final[prediction_idx][0] += models_probas[model_idx][prediction_idx][0] * models_recall_scores[
                model_idx]
            proba_final[prediction_idx][1] += models_probas[model_idx][prediction_idx][1] * models_precision_scores[
                model_idx]
        proba_final[prediction_idx][0] /= models_recall_scores.sum()
        proba_final[prediction_idx][1] /= models_precision_scores.sum()
    predictions_final = proba_final.argmax(axis=1)
    predictions_final = np.vectorize(lambda label: labels[label])(predictions_final)
    return predictions_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--df-path',
                        default=os.path.abspath("./data/normalized/test_normalized.tsv"),
                        required=False,
                        type=str,
                        help='dataframe path')
    parser.add_argument('--output-path',
                        default='./predictions.txt',
                        type=str,
                        required=False,
                        help='device type')
    args = parser.parse_args()
    model_paths = [
        '/content/drive/My Drive/Projects/covid19tweet/releases/models/digitalepidemiologylab+covid-twitter-bert_1_1_16_2e-05_2_380343_9202_9280.bin',
        '/content/drive/My Drive/Projects/covid19tweet/releases/models/digitalepidemiologylab+covid-twitter-bert_1_1_16_2e-05_3_1_9042_9407.bin']
    df_path = 'https://raw.githubusercontent.com/VinAIResearch/COVID19Tweet/master/valid.tsv'
    predictions = ensemble_predict(model_paths, df_path, labels=['UNINFORMATIVE', 'INFORMATIVE'])
    np.savetxt(args.output_path, predictions, delimiter="\n", fmt="%s")

