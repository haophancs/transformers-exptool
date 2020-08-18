import numpy as np
from utils.modeling import bert_clf

if __name__ == '__main__':
    bert_clf.predict(pretrained_bert_name='digitalepidemiologylab/covid-twitter-bert',
                     batch_size=32,
                     learning_rate=2e-5,
                     epochs=4,
                     random_state=380343)
