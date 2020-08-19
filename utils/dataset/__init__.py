import multiprocessing
import os
import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils.dataset.embedding import last_layer_features, text_vector
from utils.pretrained import load_pretrained_tokenization
from utils import __dataset_path__

dataset_prefix = os.path.join(__dataset_path__, 'embedded')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BertDataset(Dataset):

    @staticmethod
    def from_df(df_path, pretrained_bert_name):
        test_df = pd.read_csv(df_path,
                              sep='\t',
                              header=None)  # check
        tokenizer, encode_config = load_pretrained_tokenization(pretrained_bert_name)

        with_labels = 1 in test_df.columns
        if with_labels:
            dataset = BertDataset(texts=test_df[0].values,
                                  labels=test_df[1].values,
                                  tokenizer=tokenizer,
                                  encode_config=encode_config)
        else:
            dataset = BertDataset(texts=test_df[0].values,
                                  tokenizer=tokenizer,
                                  encode_config=encode_config,
                                  labels=None)
        return dataset

    def __init__(self, texts, tokenizer, encode_config, labels=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.encode_config = encode_config
        self.texts = np.vectorize(lambda txt: str(txt))(texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        if self.labels is not None:
            label = self.labels[item]
        else:
            label = None
        encoded = self.tokenizer.encode_plus(
            text,
            **self.encode_config
        )
        dataset = dict()
        dataset['text'] = text
        dataset['input_ids'] = encoded['input_ids'].flatten()
        dataset['attention_mask'] = encoded['attention_mask'].flatten()
        if label is not None:
            dataset['label'] = torch.tensor(label, dtype=torch.long)
        return dataset


class BertEmbeddedDataset(Dataset):
    def _generate_embedded(self, text):
        embedded = None
        if self.embedding_type == 'last-layer':
            embedded = last_layer_features(text,
                                           bert_model=self.model,
                                           bert_tokenizer=self.tokenizer,
                                           encode_config=self.encode_config)
        elif self.embedding_type == 'text-vector1d':
            embedded = text_vector(text,
                                   bert_model=self.model,
                                   bert_tokenizer=self.tokenizer,
                                   encode_config=self.encode_config)
        return embedded

    def __init__(self, embedding_type=None, texts=None, embedded_features=None, labels=None, pretrained_bert_name=""):
        assert embedding_type in ['last-layer', 'text-vector1d']
        self.embedding_type = embedding_type
        self.texts = texts
        self.labels = labels
        self.tokenizer = None
        self.encode_config = None
        self.model = None
        self.embedded_features = embedded_features
        self.model_name = pretrained_bert_name

    def start_embedding(self, tokenizer, encode_config, model, device=device):
        self.tokenizer = tokenizer
        self.encode_config = encode_config
        self.model = model
        self.embedded_features = torch.tensor(np.stack(np.vectorize(lambda txt: self._generate_embedded(txt),
                                                                    otypes=[object])(self.texts)))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        embedded = self.embedded_features[item]
        dataset = {
            'text': text,
            'embedded': embedded,
            'label': torch.tensor(label, dtype=torch.long)
        }
        return dataset

    def dump_to_file(self, name):
        embedded = self.embedded_features.detach().cpu().numpy()
        dict_to_save = {
            "texts": self.texts,
            "embedded_features": embedded,
            "labels": self.labels
        }
        path = os.path.join(__dataset_path__, f'embedded/{name}_{self.embedding_type}_{self.model_name}')
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(dict_to_save, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_from_file(embedding_type, pretrained_bert_name, name):
        path = os.path.join(__dataset_path__, f'embedded/{name}_{embedding_type}_{pretrained_bert_name}')
        with open(path + '.pkl', 'rb') as fp:
            data = pickle.load(fp)
            return BertEmbeddedDataset(embedding_type=embedding_type,
                                       texts=data["texts"],
                                       embedded_features=torch.tensor(data["embedded_features"]),
                                       labels=data["labels"],
                                       pretrained_bert_name=pretrained_bert_name)


def create_data_loader(dataset, batch_size, shuffle=True, num_workers=None):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
