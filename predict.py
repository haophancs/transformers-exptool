import argparse
from utils.pretrained import PretrainedOptionsAvailable
from utils.modeling import bert_clf
import numpy as np
import os
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
label_map = {0: 'UNINFORMATIVE', 1: 'INFORMATIVE'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained-bert',
                        default="digitalepidemiologylab/covid-twitter-bert",
                        choices=PretrainedOptionsAvailable,
                        required=False,
                        type=str,
                        help='name of pretrained bert')
    parser.add_argument('--model-path',
                        required=True,
                        type=str,
                        help='model path')
    parser.add_argument('--df-path',
                        default=os.path.abspath("./data/normalized/test_normalized.tsv"),
                        required=False,
                        type=str,
                        help='dataframe path')
    parser.add_argument('--batch-size',
                        default=16,
                        required=False,
                        type=int,
                        help='value of batch size')
    parser.add_argument('--random-state',
                        default=42,
                        required=False,
                        type=int,
                        help='random state')
    parser.add_argument('--map-label',
                        default=True,
                        required=False,
                        type=bool,
                        help='label mapping')
    parser.add_argument('--device',
                        default=device,
                        type=str,
                        required=False,
                        help='device type')
    args = parser.parse_args()
    texts, predictions, predictions_proba, real_labels = bert_clf.predict(pretrained_bert_name=args.pretrained_bert,
                                                                          model_path=args.model_path,
                                                                          batch_size=args.batch_size,
                                                                          random_state=args.random_state,
                                                                          df_path=args.df_path,
                                                                    device=torch.device(device))
    pred_file = "predictions.txt"
    if args.map_label:
        predictions = np.vectorize(lambda label: label_map[label])(predictions)
    np.savetxt(pred_file, predictions, delimiter="\n", fmt="%s")
    pred_prob_file = "predictions_proba"
    np.save(pred_prob_file, predictions_proba)
