from utils.inference import *
import numpy as np
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--df-path',
                        default=os.path.abspath("./data/raw/test.tsv"),
                        required=False,
                        type=str,
                        help='dataframe path')
    parser.add_argument('--output-path',
                        default='./predictions.txt',
                        type=str,
                        required=False,
                        help='output path')
    parser.add_argument('--model-path',
                        default='',
                        type=str,
                        help='model path')
    parser.add_argument('--model-list-path',
                        default='./model_list.txt',
                        type=str,
                        help='path of model list')
    parser.add_argument('--voting-type',
                        default='',
                        choices=['hard', 'soft'],
                        type=str,
                        help='type of voting for ensemble method')
    parser.add_argument('--run-eval', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']))

    args = parser.parse_args()
    if not args.model_path:
        assert args.voting_type == ''
        predictions = single_predict(args.model_path, args.df_path, labels=['UNINFORMATIVE', 'INFORMATIVE'])
    else:
        if args.voting_type == '':
            args.voting_type = 'hard'
        with open(args.model_list_path) as f:
            model_paths = [line.rstrip() for line in f]
        if args.voting_type == 'soft':
            predictions = soft_voting_predict(model_paths, args.df_path, labels=['UNINFORMATIVE', 'INFORMATIVE'])
        else:
            predictions = hard_voting_predict(model_paths, args.df_path, labels=['UNINFORMATIVE', 'INFORMATIVE'])
    np.savetxt(args.output_path, predictions, delimiter="\n", fmt="%s")
    if args.run_eval:
        os.system(f'python3 evaluator.py {args.output_path} {args.df_path}')
