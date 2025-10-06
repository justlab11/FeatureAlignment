#!/bin/bash
set -e

pip install --upgrade pip
pip install -r requirements.txt

mkdir -p data

FILEID="0B4IapRTv9pJ1WGZVd1VDMmhwdlE"
FILENAME="domain_adaptation_images.tar.gz"
wget --no-check-certificate "https://docs.google.com/uc?export=download&id=${FILEID}" -O ${FILENAME}

tar -xvzf domain_adaptation_images.tar.gz -C data
rm domain_adaptation_images.tar.gz

# python .\run_main.py --config_fname="configs/META_config.yaml"