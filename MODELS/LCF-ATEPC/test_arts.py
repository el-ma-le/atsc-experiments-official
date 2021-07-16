import os

dataset = "laptop" # arts-rest, arts-lap

model_dict_lap = {}
with open("lcf-atepc_lap_model_dict.txt") as f:
    for line in f:
        (key, val) = line.split()
        model_dict_lap[key] = val

#model_dict_rest = {}
#with open("lcf-atepc_rest_model_dict.txt") as f:
#    for line in f:
#        (key, val) = line.split()
#        model_dict_rest[key] = val


for model_name, model in model_dict_lap.items():
# for model_name, model in model_dict_rest.items():

    run_name = model_name+"_test-on-"+dataset

    print('>' * 100)
    print("Running {} data on model {}...".format(dataset, model_name))
             
    command = "python test.py --dataset %s --model %s --run_name %s " % (
                dataset, model, run_name)
    os.system(command)