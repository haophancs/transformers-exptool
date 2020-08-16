import logging
logging.basicConfig(level=logging.ERROR)

from utils.dataset.normalization import normalize_series
from utils.pretrained import PretrainedOptionsAvailable
import pandas as pd
import argparse
import os

label_map = {"UNINFORMATIVE": 0, "INFORMATIVE": 1}

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path',
                        default="./data/raw/train.tsv",
                        required=False,
                        type=str,
                        help='path to the train set')
    parser.add_argument('--valid-path',
                        default="./data/raw/valid.tsv",
                        required=False,
                        type=str,
                        help='path to the valid set')
    parser.add_argument('--test-path',
                        default="./data/raw/test.tsv",
                        required=False,
                        type=str,
                        help='path to the test set')
    parser.add_argument('--embedding-type',
                        default="last-layer",
                        choices=['last-layer', 'text-vector1d'],
                        required=False,
                        type=str,
                        help='type of embedding')
    parser.add_argument('--pretrained-bert',
                        default="bert-base-cased",
                        choices=PretrainedOptionsAvailable,
                        required=False,
                        type=str,
                        help='name of pretrained bert')
    parser.add_argument('--embedded-dir',
                        default="./data/embedded",
                        required=False,
                        type=str,
                        help='directory to save embedded data')
    parser.add_argument('--normalized-dir',
                        default="./data/normalized",
                        required=False,
                        type=str,
                        help='directory to save normalized text data')
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
    args = parser.parse_args()
    print('Loading dataset...', end=' ')
    train_df = pd.read_csv(args.train_path, sep='\t', header=None)[:args.train_limit].drop(columns=[0])
    valid_df = pd.read_csv(args.valid_path, sep='\t', header=None)[:args.valid_limit].drop(columns=[0])
    test_df = pd.read_csv(args.test_path, sep='\t', header=None)[:args.test_limit].drop(columns=[0])
    print('done')

    print('Normalizing texts...', end=' ')
    train_df[1] = normalize_series(train_df[1])
    valid_df[1] = normalize_series(valid_df[1])
    test_df[1] = normalize_series(test_df[1])
    print('done')

    print('Mapping label...', end=' ')
    train_df[2] = train_df[2].apply(lambda label: label_map[label])
    valid_df[2] = valid_df[2].apply(lambda label: label_map[label])
    test_df[2] = test_df[2].apply(lambda label: label_map[label])
    print('done')

    train_df.to_csv(os.path.join(args.normalized_dir, 'train_normalized.tsv'), sep='\t', index=False, header=False)
    valid_df.to_csv(os.path.join(args.normalized_dir, 'valid_normalized.tsv'), sep='\t', index=False, header=False)
    test_df.to_csv(os.path.join(args.normalized_dir, 'test_normalized.tsv'), sep='\t', index=False, header=False)
    print('Normalized dataset saved at', os.path.abspath(args.normalized_dir))
'''
    y_train = torch.tensor(train_df[2].values, dtype=torch.float32)
    y_valid = torch.tensor(valid_df[2].values, dtype=torch.float32)
    y_test = torch.tensor(test_df[2].values, dtype=torch.float32)
    torch.save(y_train, os.path.join(args.embedded_dir, 'y_train.pt'))
    torch.save(y_valid, os.path.join(args.embedded_dir, 'y_valid.pt'))
    torch.save(y_test, os.path.join(args.embedded_dir, 'y_test.pt'))

    print('Loading pretrained bert model and tokenizer...', end=' ')
    embedding_type = args.embedding_type
    pretrained_bert = args.pretrained_bert
    model, tokenizer = load_pretrained_bert(pretrained_bert)
    print('done')

    print(f'Extracting features ({embedding_type}) of train set...', end=' ')
    X_train = series_extract_features(train_df[1],
                                      option=embedding_type,
                                      bert_tokenizer=tokenizer,
                                      bert_model=model,
                                      return_tensors='pt')
    print('done')
    print(f'Extracting features ({embedding_type}) of valid set...', end=' ')
    X_valid = series_extract_features(valid_df[1],
                                      option=embedding_type,
                                      bert_tokenizer=tokenizer,
                                      bert_model=model,
                                      return_tensors='pt')
    print('done')
    print(f'Extracting features ({embedding_type}) of test set...', end=' ')
    X_test = series_extract_features(test_df[1],
                                     option=embedding_type,
                                     bert_tokenizer=tokenizer,
                                     bert_model=model,
                                     return_tensors='pt')
    print('done')
    torch.save(X_train, os.path.join(args.embedded_dir, f"X_train_{pretrained_bert}_{embedding_type}.pt"))
    torch.save(X_valid, os.path.join(args.embedded_dir, f"X_valid_{pretrained_bert}_{embedding_type}.pt"))
    torch.save(X_test, os.path.join(args.embedded_dir, f"X_test_{pretrained_bert}_{embedding_type}.pt"))
    print('Feature tensors saved at', os.path.abspath(args.embedded_dir))
'''
