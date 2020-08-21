from utils.inference import *
import numpy as np
import argparse
import os

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
                        help='output path')
    parser.add_argument('--voting-type',
                        default='hard',
                        choices=['hard', 'soft'],
                        type=str,
                        help='type of voting for ensemble method')
    args = parser.parse_args()
    model_paths = [
        '/content/drive/My Drive/Projects/covid19tweet/releases/models/digitalepidemiologylab+covid-twitter-bert_1_1_16_2e-05_1_96_9198_9237.bin',
        '/content/drive/My Drive/Projects/covid19tweet/releases/models/digitalepidemiologylab+covid-twitter-bert_1_1_16_2e-05_2_144_9059_9386.bin',
        '/content/drive/My Drive/Projects/covid19tweet/releases/models/digitalepidemiologylab+covid-twitter-bert_1_1_16_2e-05_2_380343_9202_9280.bin',
        '/content/drive/My Drive/Projects/covid19tweet/releases/models/digitalepidemiologylab+covid-twitter-bert_1_1_16_2e-05_3_1_9042_9407.bin',
        '/content/drive/My Drive/Projects/covid19tweet/releases/models/digitalepidemiologylab+covid-twitter-bert_1_1_16_2e-05_3_25_9236_9216.bin',
        '/content/drive/My Drive/Projects/covid19tweet/releases/models/digitalepidemiologylab+covid-twitter-bert_1_1_16_2e-05_4_747_9076_9364.bin',
        '/content/drive/My Drive/Projects/covid19tweet/releases/models/digitalepidemiologylab+covid-twitter-bert_1_1_16_3e-05_2_380343_9216_9216.bin']
    if args.voting_type == 'soft':
        predictions = soft_voting_predict(model_paths, args.df_path, labels=['UNINFORMATIVE', 'INFORMATIVE'])
    else:
        predictions = hard_voting_predict(model_paths, args.df_path, labels=['UNINFORMATIVE', 'INFORMATIVE'])
    np.savetxt(args.output_path, predictions, delimiter="\n", fmt="%s")
