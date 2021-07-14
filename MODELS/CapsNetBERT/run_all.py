import yaml
import os
from train.train import train
from train.test import test
from data_process.data_process import data_process
import mlflow

base_path_dict = {"laptop": "/home/ubuntu/mylrz/atsc-experiments/DATA/SEMEVAL-14-LAP",
                    "restaurant": "/home/ubuntu/mylrz/atsc-experiments/DATA/SEMEVAL-14-REST",
                    "mams": "/home/ubuntu/mylrz/atsc-experiments/DATA/MAMS"}

for dataset in ["laptop", "restaurant", "mams"]:

    for seed in [1,2,3,4,5]:

        if dataset != "mams":

            for split in [1,2,3,4,5]:

                print('>' * 100)
                print("Running dataset {} with seed {} and split {}...".format(dataset, seed, split))

                mlflow.set_tracking_uri("/home/ubuntu/mylrz/atsc-experiments/MODELS/mlruns")
                mlflow.set_experiment("capsnetbert")

                config = yaml.safe_load(open('all_configs.yml'))
                config['rand_state'] = split
                config['seed'] = seed
                config['base_path'] = base_path_dict[dataset]

                mode = config['mode']
                os.environ["CUDA_VISIBLE_DEVICES"] = str(config['aspect_' + mode + '_model'][config['aspect_' + mode + '_model']['type']]['gpu'])

                with mlflow.start_run():    

                    mlflow.log_artifact("all_configs.yml")
                    mlflow.log_params({"seed": config['seed'], "split": config['rand_state'], "dataset": config['base_path']})

                    data_process(config)

                    model, best_model_dict = train(config)
                    test(config, model, best_model_dict)

        else:
            print('>' * 100)
            print("Running dataset {} with seed {}...".format(dataset, seed))

            mlflow.set_tracking_uri("/home/ubuntu/mylrz/atsc-experiments/MODELS/mlruns")
            mlflow.set_experiment("capsnetbert")

            config = yaml.safe_load(open('all_configs.yml'))
            config['rand_state'] = 0
            config['seed'] = seed
            config['base_path'] = base_path_dict[dataset]

            mode = config['mode']
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config['aspect_' + mode + '_model'][config['aspect_' + mode + '_model']['type']]['gpu'])

            with mlflow.start_run():    

                mlflow.log_artifact("all_configs.yml")
                mlflow.log_params({"seed": config['seed'], "split": config['rand_state'], "dataset": config['base_path']})

                data_process(config)

                model, best_model_dict = train(config)
                test(config, model, best_model_dict)