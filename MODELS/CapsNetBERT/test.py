import yaml
import os
from train.test_arts import test
import mlflow
from data_process.data_process import data_process_arts

base_path_dict = {"lap": "../../DATA/ARTS-LAP",
                    "rest": "../../DATA/ARTS-REST"}

model_dict_lap = {}
with open("capsnetbert_lap_model_dict.txt") as f:
    for line in f:
        (key, val) = line.split()
        model_dict_lap[key] = val

#model_dict_rest = {}
#with open("capsnetbert_rest_model_dict.txt") as f:
#    for line in f:
#        (key, val) = line.split()
#        model_dict_rest[key] = val


testset = "lap" # "rest"

config = yaml.safe_load(open('test_config.yml'))
config['base_path'] = base_path_dict[testset]

mode = config['mode']
os.environ["CUDA_VISIBLE_DEVICES"] = str(config['aspect_' + mode + '_model'][config['aspect_' + mode + '_model']['type']]['gpu'])

data_process_arts(config)

for model_name, model in model_dict_lap.items():
# for model_name, model in model_dict_rest.items():

    print('>' * 100)
    print("Running arts-{} data on model {}...".format(testset, model_name))

    mlflow.set_tracking_uri("/home/ubuntu/mylrz/atsc-experiments/MODELS/mlruns")
    mlflow.set_experiment("capsnetbert")
    run_name = model_name+"_test-on-arts-"+testset

    with mlflow.start_run(run_name=run_name):

        mlflow.log_artifact("test_config.yml")
        test(config, model)