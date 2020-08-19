import logging
logging.basicConfig(level=logging.ERROR)

from utils.dataset.normalization import normalize_series
from utils.pretrained import PretrainedOptionsAvailable
import pandas as pd
import argparse
import os
import json

label_map = {"UNINFORMATIVE": 0, "INFORMATIVE": 1}


def run(pretrained_bert_name,
        train_limit=-1, valid_limit=-1, test_limit=-1,
        to_lower=None, to_ascii=None, keep_emojis=None, username=None, httpurl=None, segment_hashtag=None):
    print('Loading dataset...', end=' ')
    train_df = pd.read_csv('./data/raw/train.tsv', sep='\t', header=None).drop(columns=[0])
    valid_df = pd.read_csv('data/raw/valid.tsv', sep='\t', header=None).drop(columns=[0])
    test_df = pd.read_csv('data/raw/test.tsv', sep='\t', header=None).drop(columns=[0])
    if train_limit > 0:
        train_df = train_df[:train_limit]
    if valid_limit > 0:
        valid_df = valid_df[:valid_limit]
    if test_limit > 0:
        test_df = test_df[:test_limit]
    print('done')

    additional_config_dirpath = f'./config/bert-reconfig/{pretrained_bert_name}'
    assert pretrained_bert_name in PretrainedOptionsAvailable
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
    print('Normalizing texts...', end=' ')
    train_df[1] = normalize_series(train_df[1], to_lower=to_lower, to_ascii=to_ascii,
                                   keep_emojis=keep_emojis, segment_hashtag=segment_hashtag,
                                   username=username, httpurl=httpurl)
    valid_df[1] = normalize_series(valid_df[1], to_lower=to_lower, to_ascii=to_ascii,
                                   keep_emojis=keep_emojis, segment_hashtag=segment_hashtag,
                                   username=username, httpurl=httpurl)
    test_df[1] = normalize_series(test_df[1], to_lower=to_lower, to_ascii=to_ascii,
                                  keep_emojis=keep_emojis, segment_hashtag=segment_hashtag,
                                  username=username, httpurl=httpurl)
    print('done')

    print('Mapping label...', end=' ')
    train_df[2] = train_df[2].apply(lambda label: label_map[label])
    valid_df[2] = valid_df[2].apply(lambda label: label_map[label])
    if 2 in test_df.columns:
        test_df[2] = test_df[2].apply(lambda label: label_map[label])
    print('done')

    train_df.to_csv(os.path.join('./data/normalized/', 'train_normalized.tsv'), sep='\t', index=False, header=False)
    valid_df.to_csv(os.path.join('./data/normalized/', 'valid_normalized.tsv'), sep='\t', index=False, header=False)
    test_df.to_csv(os.path.join('./data/normalized/', 'test_normalized.tsv'), sep='\t', index=False, header=False)
    print('Normalized dataset saved at', os.path.abspath('./data/normalized'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained-bert',
                        default="bert-base-cased",
                        required=False,
                        choices=PretrainedOptionsAvailable,
                        type=str,
                        help='pretrained bert model name')
    parser.add_argument('--train-limit',
                        default=-1,
                        required=False,
                        type=int,
                        help='no. of rows limit of train set')
    parser.add_argument('--valid-limit',
                        default=-1,
                        required=False,
                        type=int,
                        help='no. of rows limit of valid set')
    parser.add_argument('--test-limit',
                        default=-1,
                        required=False,
                        type=int,
                        help='no. of rows limit of test set')
    parser.add_argument('--keep-emojis',
                        default=None,
                        required=False,
                        type=bool,
                        help='keep emojis')
    parser.add_argument('--to-lower',
                        default=None,
                        required=False,
                        type=bool,
                        help='to lowercased')
    parser.add_argument('--to-ascii',
                        default=None,
                        required=False,
                        type=bool,
                        help='convert to ascii')
    parser.add_argument('--segment-hashtag',
                        default=None,
                        required=False,
                        type=bool,
                        help='segment hashtag')
    args = parser.parse_args()
    run(pretrained_bert_name=args.pretrained_bert,
        train_limit=args.train_limit, valid_limit=args.valid_limit, test_limit=args.test_limit,
        to_ascii=args.to_ascii, to_lower=args.to_lower,
        keep_emojis=args.keep_emojis,
        segment_hashtag=args.segment_hashtag)
