#!/bin/bash
# build vocab for different datasets
setting=../../DATA

python prepare_vocab.py --data_dir $setting/SEMEVAL-14-REST --vocab_dir $setting/SEMEVAL-14-REST
python prepare_vocab.py --data_dir $setting/SEMEVAL-14-LAP --vocab_dir $setting/SEMEVAL-14-LAP
python prepare_vocab.py --data_dir $setting/MAMS --vocab_dir $setting/MAMS --splits_available 0
