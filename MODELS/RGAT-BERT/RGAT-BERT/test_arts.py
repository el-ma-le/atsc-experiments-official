import os

dataset = "arts-lap" # arts-lap, arts-rest

model_dict_lap = {}
with open("rgatbert_lap_model_dict.txt") as f:
    for line in f:
        (key, val) = line.split()
        model_dict_lap[key] = val

model_dict_rest = {}
with open("rgatbert_rest_model_dict.txt") as f:
    for line in f:
        (key, val) = line.split()
        model_dict_rest[key] = val

vocab_dir_dict = {"arts-lap": "/home/ubuntu/mylrz/atsc-experiments/DATA/SEMEVAL-14-LAP",
                    "arts-rest": "/home/ubuntu/mylrz/atsc-experiments/DATA/SEMEVAL-14-REST",
                    "laptop": "/home/ubuntu/mylrz/atsc-experiments/DATA/SEMEVAL-14-LAP"}
dep_dim_dict = {"mams": 80, "laptop": 100, "restaurant": 80, "arts-lap": 100, "arts-rest": 80}
att_dropout_dict = {"mams": 0.0, "laptop": 0.1, "restaurant": 0.0, "arts-lap": 0.1, "arts-rest": 0.0}

vocab_dir_path = vocab_dir_dict[dataset]
dep_dim = dep_dim_dict[dataset]
att_dropout = att_dropout_dict[dataset]

source_dir = "/home/ubuntu/mylrz/atsc-experiments/DATA"
save_dir = "saved_models"

exp_setting = "testing"
exp_dataset_dict = {"mams": "MAMS", 
                    "laptop": "SEMEVAL-14-LAP", 
                    "restaurant": "SEMEVAL-14-REST",
                    "arts-lap": "ARTS-LAP",
                    "arts-rest": "ARTS-REST"}
exp_dataset = exp_dataset_dict[dataset]
exp_path = "/".join([save_dir, exp_dataset, exp_setting])
data_dir_path = "/".join([source_dir, exp_dataset])
tee_path = exp_path + '/testing.log'

CUDA_VISIBLE_DEVICES=0 


for model_name, model in model_dict_lap.items():
#for model_name, model in model_dict_rest.items():

    print('>' * 100)
    print("Running dataset {} with model {}".format(dataset, model_name))

    run_name = model_name+"_test-on-"+dataset

    command = "python -u test.py --lr 1e-5 --num_layer 2 \
                --data_dir %s --logged_model %s --max_len 90 \
                --vocab_dir %s --run_name %s --dep_dim %s \
                --save_dir exp_path --model 'RGAT' \
                --output_merge 'gate' --reset_pool 2>&1 | tee tee_path " % (
        data_dir_path, model, vocab_dir_path, run_name, dep_dim)
    os.system(command)