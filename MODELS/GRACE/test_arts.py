import os

DATA_DIR = "../../DATA/"
CUDA_VISIBLE_DEVICES = 0 

dataset = "arts-lap" # "arts-lap", "arts-rest"

model_dict_lap = {}
with open("grace_lap_model_dict.txt") as f:
    for line in f:
        (key, val) = line.split()
        model_dict_lap[key] = val

model_dict_rest = {}
with open("grace_rest_model_dict.txt") as f:
    for line in f:
        (key, val) = line.split()
        model_dict_rest[key] = val


for model_name, model in model_dict_lap.items():
#for model_name, model in model_dict_rest.items():
    
    run_name = model_name+"_test-on-"+dataset
    split = int(model_name.split("_")[1][-1])

    print ('>' * 100)
    print(">> Running model {} on dataset {}".format(model_name, dataset))

    command_3 = "python test.py --do_eval --data_dir %s --data_name %s \
                --bert_model=bert-base-uncased --do_lower_case --max_seq_length=128 \
                --warmup_proportion=0.1 --use_ghl \
                --run_name %s --logged_model %s --split %s " % (DATA_DIR, dataset, 
                run_name, model, split)

    os.system(command_3)