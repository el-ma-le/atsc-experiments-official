# atsc-experiments-official

## Used Datasets
- SemEval-14 Restaurants 
- SemEval-14 Laptops
- MAMS (Restaurants)
- ARTS Restaurants
- ARTS Laptops

For data preprocessing refer to the corresponding folder where data sources are indicated and preprocessing scripts are provided.

## Model Sources
- CapsNetBERT: https://github.com/siat-nlp/MAMS-for-ABSA
- MGATN in the model collection repository ABSA-PYTORCH: https://github.com/songyouwei/ABSA-PyTorch
- RGAT-BERT: https://github.com/muyeby/RGAT-ABSA
- BERT-E2E-ABSA: https://github.com/lixin4ever/BERT-E2E-ABSA
- GRACE: https://github.com/ArrowLuo/GRACE
- LCF-ATEPC: https://github.com/yangheng95/LCF-ATEPC

## Prerequisites
- CapsNetBERT: download glove from http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip and save in `/DATA/`, `python -m spacy download en`
- MGATN: download glove from https://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip and save in `/DATA/`
- RGAT-BERT: download glove from http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip and save in `/DATA/`
- GRACE: download pretrained weights from https://drive.google.com/u/0/uc?id=1xYD3bl1qAiJlK4Ruyr7MWsgBLDGk0YjT and save in `/GRACE/pretrained_weight/`

## Required python package versions
A requirements file can be found in each model folder.

## Train & Test Files
For test runs, choose a logged model and add it to the corresponding file or create model dicts. For ARTS, models trained on the corresponding SemEval-14 datasets are taken.
- CapsNetBERT: 
  - for training and testing, select data and hyperparameters in `config.yml`, `python train_.py`, 
  - for all runs at once, select hyperparameters in `all_configs.yml` and run: `run_all.py`
  - to test arts data: `python test.py`
- MGATN: 
  - `python train.py --dataset <laptop, restaurant, mams> --train_seed <seed_number 1,2,3,4,5 in case of semeval data> --seed <1,2,3,4,5>`, 
  - `python test.py --dataset <laptop, restaurant, mams>`
  - for all runs (per dataset) at once: `python run_all.py`
  - to test arts data: `python test_arts.py`
- RGAT-BERT: 
  - run once: `bash build_vocab.sh`, 
  - `bash run-<dataset-name>-BERT.sh`, 
  - `python test.py --vocab_dir <vocab_dir of trained model>`
  - for all runs (per dataset) at once: `python run_all.py`
  - to test arts data: `python test_arts.py`
- BERT-E2E-ABSA: 
  - `python train.py`, 
  - `sh test.sh`
  - for all runs at once: `python run_all.py`
  - to test arts data: `python test_arts.py`
- GRACE:
  - set hyperparameters (numbers behind dataset indicate split): 
    `DATA_NAME=lap1
    VAT_NUM=1
    PRETRAINED_WEIGHT=pretrained_weight/pytorch_model.bin
    DATA_DIR=/home/ubuntu/mylrz/atsc-experiments/DATA/`
  - ATE - with GHL: `CUDA_VISIBLE_DEVICES=0 python ate_run.py --do_train --do_eval --data_dir=${DATA_DIR} --data_name=${DATA_NAME} --output_dir=out_${DATA_NAME} --bert_model=bert-base-uncased --do_lower_case --max_seq_length=128 --warmup_proportion=0.1 --train_batch_size 32 --num_train_epochs 5 --learning_rate 3e-5 --use_ghl --init_model ${PRETRAINED_WEIGHT} --train_seed=${TRAIN_SEED}`
  - ATE - with GHL + VAT: `CUDA_VISIBLE_DEVICES=0 python ate_run.py --do_train --do_eval --data_dir=${DATA_DIR} --data_name=${DATA_NAME} --output_dir=out_${DATA_NAME} --bert_model=bert-base-uncased --do_lower_case --max_seq_length=128 --warmup_proportion=0.1 --train_batch_size 32 --num_train_epochs ${VAT_NUM} --learning_rate 1e-5 --use_ghl --use_vat --init_model out_${DATA_NAME}/pytorch_model.bin.4 --train_seed=${TRAIN_SEED}`
  - ATE + ASC - with GHL + VAT: `CUDA_VISIBLE_DEVICES=0 python ate_asc_run.py --do_train --do_eval --data_dir=${DATA_DIR} --data_name=${DATA_NAME} --output_dir=out_${DATA_NAME}_ateacs --bert_model=bert-base-uncased --do_lower_case --max_seq_length=128 --warmup_proportion=0.1 --train_batch_size 32 --num_train_epochs 10 --learning_rate 3e-6 --use_ghl --init_model out_${DATA_NAME}/pytorch_model.bin.$((VAT_NUM-1)) --patience --train_seed=${TRAIN_SEED}`
  - for all runs at once: `python run_all.py`
  - to test arts data: `python test_arts.py`
- LCF-ATEPC:
  - `python train.py`
  - for all runs at once: `python run_all.py`
  - to test arts data: `python test_arts.py`
