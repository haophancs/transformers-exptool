import argparse
from utils.pretrained import PretrainedOptionsAvailable
from utils.modeling import bert_clf
import prepare_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained-bert',
                        default="bert-base-cased",
                        choices=PretrainedOptionsAvailable,
                        required=False,
                        type=str,
                        help='name of pretrained bert')
    parser.add_argument('--batch-size',
                        default=16,
                        required=False,
                        type=int,
                        help='value of batch size')
    parser.add_argument('--epochs',
                        default=4,
                        required=False,
                        type=int,
                        help='number of epoch(s)')
    parser.add_argument('--learning-rate',
                        default=2e-5,
                        required=False,
                        type=float,
                        help='value of learning rate')
    parser.add_argument('--random-state',
                        default=42,
                        required=False,
                        type=int,
                        help='random state')
    args = parser.parse_args()
    # prepare_data.run(pretrained_bert_name=args.pretrained_bert)
    bert_clf.train(pretrained_bert_name=args.pretrained_bert,
                   batch_size=args.batch_size,
                   epochs=args.epochs,
                   learning_rate=args.learning_rate,
                   random_state=args.random_state)
    bert_clf.eval(pretrained_bert=args.pretrained_bert,
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  learning_rate=args.learning_rate,
                  random_state=args.random_state)
