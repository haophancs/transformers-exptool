#!/bin/sh
sudo add-apt-repository ppa:openjdk-r/ppa
sudo apt-get update
sudo apt-get install openjdk-8-jre openjdk-8-jdk
pip install -r requirements.txt

mkdir -p models/pretrained/vinai
mkdir data/normalized
mkdir data/embedded

wget https://public.vinai.io/BERTweet_base_transformers.tar.gz
tar -xzvf BERTweet_base_transformers.tar.gz -C ./models/pretrained/vinai
rm BERTweet_base_transformers.tar.gz

mv models/pretrained/vinai/BERTweet_base_transformers/ models/pretrained/vinai/bertweet-base
rm models/pretrained/vinai/bertweet-base/model.bin
rm models/pretrained/vinai/bertweet-base/config.json

