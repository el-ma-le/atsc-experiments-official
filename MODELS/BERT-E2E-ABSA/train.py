import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed = 1 #1,2,3,4,5
train_seed = 1 #1,2,3,4,5
model_type = 'bert'
absa_type = 'tfm'
tfm_mode = 'finetune'
fix_tfm = 0
task_name = 'laptop14' # rest14, mams, laptop14
data_dir_dict = {'laptop14': 'SEMEVAL-14-LAP',
                 'rest14': 'SEMEVAL-14-REST',
                 'mams': 'MAMS'}
data_dir = data_dir_dict[task_name]
warmup_steps = 0
overfit = 0
num_epochs = 50
patience = 10

if task_name == 'laptop14' or task_name == 'mams':
    train_batch_size = 32
elif task_name == 'rest14':
    train_batch_size = 16
else:
    raise Exception("Unsupported dataset %s!!!" % task_name)

command = "python main.py --model_type %s --absa_type %s --tfm_mode %s --fix_tfm %s " \
            "--model_name_or_path bert-base-uncased --data_dir ../../DATA/%s --task_name %s " \
            "--per_gpu_train_batch_size %s --per_gpu_eval_batch_size 8 --learning_rate 2e-5 " \
            "--num_train_epochs %s --warmup_steps %s --do_train --do_eval --do_lower_case " \
            "--seed %s --train_seed %s --tagging_schema BIEOS --overfit %s --patience %s " \
            "--overwrite_output_dir --eval_all_checkpoints --MASTER_ADDR localhost --MASTER_PORT 28512 " % (
    model_type, absa_type, tfm_mode, fix_tfm, data_dir, task_name, train_batch_size, num_epochs, warmup_steps, seed, train_seed, overfit, patience)

os.system(command)