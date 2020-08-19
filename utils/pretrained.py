import argparse
import os
import json
from utils import __models_path__, __config_path__

import torch
from fairseq.data import Dictionary
from fairseq.data.encoders.fastbpe import fastBPE
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import RobertaConfig as RobertaConfig
from transformers import RobertaModel as RobertaModel
from typing import List

PretrainedOptionsAvailable = ['bert-base-cased', 'vinai/bertweet-base', 'albert-base-v2', 'xlm-roberta-large',
                              'digitalepidemiologylab/covid-twitter-bert', 'xlnet-large-cased', 'albert-xlarge-v2',
                              'bert-large-uncased`']

pretrained_prefix = os.path.abspath(os.path.join(__models_path__, 'pretrained'))
parser = argparse.ArgumentParser()
parser.add_argument('--bpe-codes',
                    default="/Absolute-path-to/bertweet-base/bpe.codes",
                    required=False,
                    type=str,
                    help='path to fastBPE BPE')
args = parser.parse_args(
    ['--bpe-codes', os.path.join(pretrained_prefix, 'vinai/bertweet-base/bpe.codes')])


def BERTweetConfig(additional_config):
    config = RobertaConfig.from_pretrained(
        os.path.join(pretrained_prefix, "vinai/bertweet-base/org_config.json"),
        **additional_config
    )
    return config


def BERTweetModel(additional_config):
    model = RobertaModel.from_pretrained(
        os.path.join(pretrained_prefix, "vinai/bertweet-base/pytorch_model.bin"),
        config=BERTweetConfig(additional_config)
    )
    return model


class BERTweetTokenizer:
    def __init__(self):
        self._bpe = fastBPE(args)
        self._vocab = Dictionary()
        self._vocab.add_from_file(os.path.join(pretrained_prefix, "vinai/bertweet-base/dict.txt"))

    @staticmethod
    def _pad_input_ids(input_ids, max_length):
        padded_input_ids = input_ids[:max_length]
        padded_input_ids = padded_input_ids + [0] * (max_length - len(padded_input_ids))
        return padded_input_ids

    @staticmethod
    def _create_attention_mask(input_ids, max_length):
        attention_mask = [1] * len(input_ids) + [0] * (max_length - len(input_ids))
        return attention_mask

    def encode(self, text: str, append_eos=False) -> List[str]:
        subwords = '<s> ' + self._bpe.encode(text) + ' </s>'
        input_ids = self._vocab.encode_line(subwords, append_eos=append_eos, add_if_not_exist=False).long().tolist()
        return input_ids

    def encode_plus(self,
                    text,
                    **kwargs):
        append_eos = kwargs.get('append_eos', False)
        max_length = kwargs.get('max_length', 128)
        pad_to_max_length = kwargs.get('pad_to_max_length', False)
        return_attention_mask = kwargs.get('return_attention_mask', False)
        return_tensors = kwargs.get('return_tensors', 'pt')
        truncation = kwargs.get('truncation', False)

        input_ids = self.encode(text, append_eos=append_eos)
        if max_length is not None:
            if truncation:
                # if len(input_ids) > max_length:
                #    print(f'Truncated: max sequence length is {max_length}, got {len(input_ids)}')
                input_ids = input_ids[:max_length]
            else:
                print("Truncation was not explicitely activated but `max_length` is provided a specific value, "
                      "please use `truncation=True` to explicitely truncate examples to max length. Defaulting to "
                      "'only_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the "
                      "tokenizer you may want to check this is the right behavior.")
                max_length = len(input_ids)
        encoding = {'input_ids': input_ids}
        if return_attention_mask:
            encoding['attention_mask'] = self._create_attention_mask(encoding['input_ids'], max_length)
        if pad_to_max_length:
            encoding['input_ids'] = self._pad_input_ids(encoding['input_ids'], max_length)
        if return_tensors == 'pt':
            for key, value in encoding.items():
                encoding[key] = torch.tensor([value], dtype=torch.long)
        return encoding


def load_pretrained_tokenization(pretrained_name):
    additional_config_dirpath = os.path.join(__config_path__, f'bert-reconfig/{pretrained_name}')
    assert pretrained_name in PretrainedOptionsAvailable
    with open(os.path.join(additional_config_dirpath, 'tokenizer.json')) as JSON:
        encode_config = json.loads(JSON.read())
    if pretrained_name != 'vinai/bertweet-base':
        bert_tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    else:
        bert_tokenizer = BERTweetTokenizer()
    return bert_tokenizer, encode_config


def load_pretrain_model_config(pretrained_name):
    additional_config_dirpath = os.path.join(__config_path__, f'bert-reconfig/{pretrained_name}')
    assert pretrained_name in PretrainedOptionsAvailable
    with open(os.path.join(additional_config_dirpath, 'model.json')) as JSON:
        additional_model_config = json.loads(JSON.read())
    bert_config = AutoConfig.from_pretrained(pretrained_name, **additional_model_config)
    return bert_config


def load_pretrained_model(pretrained_name):
    additional_config_dirpath = os.path.join(__config_path__, f'bert-reconfig/{pretrained_name}')
    assert pretrained_name in PretrainedOptionsAvailable
    with open(os.path.join(additional_config_dirpath, 'model.json')) as JSON:
        additional_model_config = json.loads(JSON.read())
    bert_config = AutoConfig.from_pretrained(pretrained_name, **additional_model_config)
    bert_model = AutoModel.from_pretrained(pretrained_name, config=bert_config)
    return bert_model
