import yaml
import os
from train.train import train
from train.test import test
from data_process.data_process import data_process
import mlflow

mlflow.set_tracking_uri("../mlruns")
mlflow.set_experiment("capsnetbert")

config = yaml.safe_load(open('config.yml'))
mode = config['mode']
os.environ["CUDA_VISIBLE_DEVICES"] = str(config['aspect_' + mode + '_model'][config['aspect_' + mode + '_model']['type']]['gpu'])

with mlflow.start_run():    

    mlflow.log_artifact("config.yml")
    data_process(config)

    model, best_model_dict = train(config)
    
    test(config, model, best_model_dict)