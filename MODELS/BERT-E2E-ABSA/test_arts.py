import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

task_name = "arts-lap" #"arts-lap" , "arts-rest"

model_type = 'bert'
absa_type = 'tfm'
data_dir_dict = {'arts-lap': 'ARTS-LAP',
                 'arts-rest': 'ARTS-REST',
                 'laptop14': 'SEMEVAL-14-LAP'}

if task_name == 'arts-lap' or task_name == 'laptop14':
    train_batch_size = 32
elif task_name == 'arts-rest':
    train_batch_size = 16
else:
    raise Exception("Unsupported dataset %s!!!" % task_name)

data_dir = data_dir_dict[task_name]

model_dict_lap = {}
with open("bert+tfm_lap_model_dict.txt") as f:
    for line in f:
        (key, val) = line.split()
        model_dict_lap[key] = val

#model_dict_rest = {}
#with open("bert+tfm_rest_model_dict.txt") as f:
#    for line in f:
#        (key, val) = line.split()
#        model_dict_rest[key] = val


for model_name, model in model_dict_lap.items():
#for model_name, model in model_dict_rest.items():

    run_name = model_name+"_test-on-"+task_name

    print('>' * 100)
    print("Running dataset {} on model {}...".format(task_name, model_name))

    command = "python test.py --model_type %s --do_lower_case " \
                "--model_name_or_path bert-base-uncased " \
                "--data_dir /home/ubuntu/mylrz/atsc-experiments/DATA/%s " \
                "--task_name %s --logged_model %s " \
                "--run_name %s --absa_type %s " % (
        model_type, data_dir, task_name, model, run_name, absa_type)

    os.system(command)