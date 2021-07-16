import os

VAT_NUM = 1
PRETRAINED_WEIGHT = "pretrained_weight/pytorch_model.bin"
DATA_DIR = "../../DATA/"
CUDA_VISIBLE_DEVICES = 0 
NUM_EPOCHS_ATSC = 50
PATIENCE = 10

datasets = ["mams"]
#datasets += ["lap{}".format(i) for i in range(1,6)]
#datasets += ["rest{}".format(i) for i in range(1,6)]

for dataset in datasets:

    for seed in [1,2,3,4,5]:

        # ATE with GHL
        output_dir = "outputs/out_"+dataset

        print('>' * 100)
        print('>' * 100)
        print("Running dataset {} with seed {}...".format(dataset, seed))

        print ('>' * 100)
        print(">> Run ATE with GHL")
        command_1 = "python ate_run.py --do_train --do_eval --data_dir %s --data_name %s \
                    --output_dir %s  --bert_model=bert-base-uncased --do_lower_case --max_seq_length=128  \
                    --warmup_proportion=0.1 --train_batch_size 32 --num_train_epochs 5 --learning_rate 3e-5 --use_ghl \
                    --init_model %s --seed %s" % (DATA_DIR, dataset, output_dir, PRETRAINED_WEIGHT, seed)

        os.system(command_1)


        # ATE with GHL + VAT
        init_model_dir = output_dir+"/pytorch_model.bin.4"

        print ('>' * 100)
        print(">> Run ATE with GHL and VAT")
        command_2 = "python ate_run.py --do_train --do_eval --data_dir %s --data_name %s \
                    --output_dir %s --bert_model=bert-base-uncased --do_lower_case --max_seq_length=128 \
                    --warmup_proportion=0.1 --train_batch_size 32 --num_train_epochs %s --learning_rate 1e-5 \
                    --use_ghl --use_vat --init_model %s --seed %s " % (DATA_DIR, dataset, output_dir, VAT_NUM, init_model_dir, seed)

        os.system(command_2)


        # ATE+ATSC with GHL + VAT
        final_output_dir = "outputs/out_"+dataset+"_ateacs"
        final_init_model_dir = "outputs/out_"+dataset+"/pytorch_model.bin."+str(VAT_NUM-1)

        print ('>' * 100)
        print(">> Run ATE + ATSC with GHL + VAT")
        command_3 = "python ate_asc_run.py --do_train --do_eval --data_dir %s --data_name %s \
                    --output_dir %s --bert_model=bert-base-uncased --do_lower_case --max_seq_length=128 \
                    --warmup_proportion=0.1 --train_batch_size 32 --num_train_epochs %s --learning_rate 3e-6 --use_ghl \
                    --init_model %s --patience %s --seed %s " % (DATA_DIR, dataset, final_output_dir, NUM_EPOCHS_ATSC, 
                    final_init_model_dir, PATIENCE, seed)

        os.system(command_3)