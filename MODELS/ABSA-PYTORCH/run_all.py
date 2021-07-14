import os

datasets = ["mams", "restaurant", "laptop"]
seeds = [1,2,3,4,5]
train_seeds = [1,2,3,4,5]

dataset_short = {"restaurant": "rest", "laptop": "lap"}

for dataset in datasets:
    for seed in seeds:
        if dataset != "mams":
            for split in train_seeds:
                print('>' * 100)
                print("Running "+dataset +" with seed "+str(seed)+" and split "+str(split))

                run_name = dataset_short[dataset]+"_split"+str(split)+"_run"+str(seed)+"_train_test-on-"+dataset_short[dataset]

                command = "python train.py --dataset %s --train_seed %s --seed %s --run_name %s " % (
                    dataset, split, seed, run_name)
                os.system(command)

        else:
            print('>' * 100)
            print("Running "+dataset +" with seed "+str(seed))

            run_name = dataset+"_run"+str(seed)+"_train_test-on-"+dataset

            command = "python train.py --dataset %s --seed %s --run_name %s " % (
                    dataset, seed, run_name)
            os.system(command)