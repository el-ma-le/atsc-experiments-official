import os

dataset = "mams" # restaurant, laptop, mams
seeds = [1,2,3,4,5]
train_seeds = [1,2,3,4,5]

att_dropout_dict = {"mams": 0.0, "laptop": 0.1, "restaurant": 0.0}
att_dropout = att_dropout_dict[dataset]
dep_dim_dict = {"mams": 80, "laptop": 100, "restaurant": 80}
dep_dim = dep_dim_dict[dataset]
num_epoch = 50
patience = 10

source_dir = "/home/ubuntu/mylrz/atsc-experiments/DATA"
save_dir = "saved_models"

exp_setting = "training"
exp_dataset_dict = {"mams": "MAMS", "laptop": "SEMEVAL-14-LAP", "restaurant": "SEMEVAL-14-REST"}
exp_dataset = exp_dataset_dict[dataset]
exp_path = "/".join([save_dir, exp_dataset, exp_setting])
data_dir_path = "/".join([source_dir, exp_dataset])
tee_path = exp_path + '/training.log'

CUDA_VISIBLE_DEVICES=0 

for seed in seeds:
    if dataset != "mams":
        for split in train_seeds:
            print('>' * 100)
            print("Running seed "+str(seed)+" split "+str(split))

            command = "python -u bert_train.py --lr 1e-5 --bert_lr 2e-5 \
                     --input_dropout 0.1 --att_dropout %s \
                     --num_layer 2 --bert_out_dim 100 --dep_dim %s --max_len 90 \
                     --data_dir %s --patience %s \
                     --vocab_dir %s --save_dir exp_path \
                     --model 'RGAT' --seed %s --train_seed %s \
                     --output_merge 'gate' --reset_pool \
                     --num_epoch %s 2>&1 | tee tee_path " % (
                att_dropout, dep_dim, data_dir_path, patience, data_dir_path, seed, split, num_epoch)
            os.system(command)

    else:
        print('>' * 100)
        print("Running seed "+str(seed))

        command = "python -u bert_train.py --lr 1e-5 --bert_lr 2e-5 \
                     --input_dropout 0.1 --att_dropout %s \
                     --num_layer 2 --bert_out_dim 100 --dep_dim %s --max_len 90 \
                     --data_dir %s --patience %s \
                     --vocab_dir %s --save_dir exp_path \
                     --model 'RGAT' --seed %s \
                     --output_merge 'gate' --reset_pool \
                     --num_epoch %s 2>&1 | tee exp_path+'/training.log' " % (
                att_dropout, dep_dim, data_dir_path, patience, data_dir_path, seed, num_epoch)
        os.system(command)