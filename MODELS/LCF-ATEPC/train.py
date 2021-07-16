# -*- coding: utf-8 -*-
# file: train.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.

import argparse
import json
import random
from sklearn.metrics import f1_score, accuracy_score, classification_report
from time import strftime, localtime

import numpy as np
import torch
import torch.nn.functional as F
from transformers.optimization import AdamW
from transformers.models.bert.modeling_bert import BertModel
from transformers import BertTokenizer
from seqeval.metrics import classification_report as seq_report
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from utils.data_utils import ATEPCProcessor, convert_examples_to_features
from model.lcf_atepc import LCF_ATEPC

import mlflow
import copy

time = '{}'.format(strftime("%y%m%d-%H%M%S", localtime()))

def evaluate(dataloader, model, label_list, args, eval_ATE=True, eval_APC=True):

    y_true = []
    y_pred = []
    n_test_correct, n_test_total = 0, 0
    test_apc_logits_all, test_polarities_all = None, None
    model.eval()
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    for input_ids_spc, input_mask, segment_ids, label_ids, polarities, valid_ids, l_mask in dataloader:
        input_ids_spc = input_ids_spc.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        valid_ids = valid_ids.to(device)
        label_ids = label_ids.to(device)
        polarities = polarities.to(device)
        l_mask = l_mask.to(device)

        with torch.no_grad():
            ate_logits, apc_logits = model(input_ids_spc, segment_ids, input_mask,
                                            valid_ids=valid_ids, polarities=polarities, attention_mask_label=l_mask)
        if eval_APC:
            polarities = model.get_batch_polarities(polarities)
            n_test_correct += (torch.argmax(apc_logits, -1) == polarities).sum().item()
            n_test_total += len(polarities)

            if test_polarities_all is None:
                test_polarities_all = polarities
                test_apc_logits_all = apc_logits
            else:
                test_polarities_all = torch.cat((test_polarities_all, polarities), dim=0)
                test_apc_logits_all = torch.cat((test_apc_logits_all, apc_logits), dim=0)

        if eval_ATE:
            if not args.use_bert_spc:
                label_ids = model.get_batch_token_labels_bert_base_indices(label_ids)
            ate_logits = torch.argmax(F.log_softmax(ate_logits, dim=2), dim=2)
            ate_logits = ate_logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()
            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                for j, m in enumerate(label):
                    if j == 0:
                        continue
                    elif label_ids[i][j] == len(label_list):
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        break
                    else:
                        temp_1.append(label_map.get(label_ids[i][j], 'O'))
                        temp_2.append(label_map.get(ate_logits[i][j], 'O'))
    if eval_APC:
        #test_acc = n_test_correct / n_test_total
        # both accuracies yield the same results
        test_acc_sk = accuracy_score(torch.argmax(test_apc_logits_all, -1).cpu(), test_polarities_all.cpu())
        test_f1_macro = f1_score(torch.argmax(test_apc_logits_all, -1).cpu(), test_polarities_all.cpu(),
                                labels=[0, 1, 2], average='macro')
        test_f1_micro = f1_score(torch.argmax(test_apc_logits_all, -1).cpu(), test_polarities_all.cpu(),
                                labels=[0, 1, 2], average='micro')
        test_f1_weighted = f1_score(torch.argmax(test_apc_logits_all, -1).cpu(), test_polarities_all.cpu(),
                                labels=[0, 1, 2], average='weighted')

        apc_result = {'apc_test_f1_macro': test_f1_macro, 'apc_test_f1_micro': test_f1_micro, 
                        'apc_test_f1_weighted': test_f1_weighted, "apc_test_acc": test_acc_sk}
    else:
        apc_result = {'apc_test_acc': 0, 'apc_test_f1_macro': 0, 'apc_test_f1_micro': 0, 'apc_test_f1_weighted': 0}

    if eval_ATE:
        report = seq_report(y_true, y_pred, digits=4)
        tmps = report.split()
        # f1 score for class ASP
        ate_result = float(tmps[7])
    else:
        ate_result = 0
        
    return apc_result, ate_result


def main(config):

    def create_dataloader(examples, label_list, max_seq_len, tokenizer, batch_size, eval=True):

        features = convert_examples_to_features(examples, label_list, max_seq_len, tokenizer)

        all_spc_input_ids = torch.tensor([f.input_ids_spc for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in features], dtype=torch.long)
        all_polarities = torch.tensor([f.polarities for f in features], dtype=torch.long)

        new_data = TensorDataset(all_spc_input_ids, all_input_mask, all_segment_ids,
                                   all_label_ids, all_polarities, all_valid_ids, all_lmask_ids)
        if eval == False:
            new_sampler = SequentialSampler(new_data)
        else:
            new_sampler = RandomSampler(new_data)

        return DataLoader(new_data, sampler=new_sampler, batch_size=batch_size)


    def train():

        print("***** Running training *****")
        print("  Num examples = %d", len(train_examples))
        print("  Batch size = %d", args.train_batch_size)
        print("  Num steps = %d", num_train_optimization_steps)
        
        max_apc_test_acc = 0
        max_apc_test_f1_macro = 0
        max_val_epoch = 0

        global_step = 0
        for epoch in range(int(args.num_train_epochs)):
            print('#' * 80)
            print('Training epoch {}'.format(epoch))
            print('#' * 80)
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                input_ids_spc, input_mask, segment_ids, label_ids, polarities, valid_ids, l_mask = batch
                loss_ate, loss_apc = model(input_ids_spc, segment_ids, input_mask, label_ids, polarities, valid_ids,
                                           l_mask)
                loss = loss_ate + loss_apc
                loss.backward()
                nb_tr_examples += input_ids_spc.size(0)
                nb_tr_steps += 1
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            # Evaluate on validation data
            apc_result, ate_result = evaluate(val_dataloader, model, label_list, args, eval_ATE=not args.use_bert_spc)
            mlflow.log_metrics({"val_accuracy": apc_result["apc_test_acc"], "val_f1_macro": apc_result["apc_test_f1_macro"]}, epoch)

            if apc_result['apc_test_acc'] >= max_apc_test_acc:
                max_apc_test_acc = apc_result['apc_test_acc']
                max_val_epoch = epoch
                best_model = copy.deepcopy(model)
                best_model_dict = copy.deepcopy(model.state_dict())
            if apc_result['apc_test_f1_macro'] >= max_apc_test_f1_macro:
                max_apc_test_f1_macro = apc_result['apc_test_f1_macro']
                max_val_epoch = epoch
                best_model = copy.deepcopy(model)
                best_model_dict = copy.deepcopy(model.state_dict())

            print('Validation Results in Epoch {}'.format(epoch))
            print(f'APC_val_acc: {apc_result["apc_test_acc"]} (max: {max_apc_test_acc})  '
                        f'APC_val_f1_macro: {apc_result["apc_test_f1_macro"]} (max: {max_apc_test_f1_macro})')
            if args.use_bert_spc:
                print(f'ATE_val_F1: {apc_result["apc_test_f1_macro"]} (max: {max_apc_test_f1_macro})'
                            f' (Unreliable since `use_bert_spc` is "True".)')
            else:
                print(f'ATE_val_f1: {ate_result} ')
            print('*' * 80)

            if epoch - max_val_epoch >= args.patience:
                print('>> early stop.')
                mlflow.log_metric("early_stopp_at_epoch", epoch)
                break

        mlflow.pytorch.log_model(best_model, "model")
        return best_model_dict


    args = config

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    processor = ATEPCProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1

    datasets = {
        'laptop': "../../DATA/SEMEVAL-14-LAP",
        'restaurant': "../../DATA/SEMEVAL-14-REST",
        'mams':  "../../DATA/MAMS"
    }
    pretrained_bert_models = {
        'laptop': "bert-base-uncased",
        'restaurant': "bert-base-uncased",
        # for loading domain-adapted BERT
        # 'restaurant': "../bert_pretrained_restaurant",
        'mams': "bert-base-uncased"
    }

    args.bert_model = pretrained_bert_models[args.dataset]
    args.data_dir = datasets[args.dataset]

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    train_examples = processor.get_train_examples(args.data_dir, args.train_seed)
    val_examples = processor.get_dev_examples(args.data_dir, args.train_seed)
    train_dataloader = create_dataloader(train_examples, label_list, args.max_seq_length, tokenizer, args.train_batch_size, eval = False)
    val_dataloader = create_dataloader(val_examples, label_list, args.max_seq_length, tokenizer,args.eval_batch_size)

    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    bert_base_model = BertModel.from_pretrained(args.bert_model)
    bert_base_model.config.num_labels = num_labels
    model = LCF_ATEPC(bert_base_model, args=args)
    model.to(device)

    for arg in vars(args):
        print('>>> {0}: {1}'.format(arg, getattr(args, arg)))
        mlflow.log_param(arg, getattr(args, arg))

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.00001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.00001}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=0.00001)

    # Train
    best_model_dict = train()

    # Evaluate on test data
    test_examples = processor.get_test_examples(args.data_dir)
    test_dataloader = create_dataloader(test_examples, label_list, args.max_seq_length, tokenizer, args.eval_batch_size)
    model.load_state_dict(best_model_dict)

    test_apc_result, test_ate_result = evaluate(test_dataloader, model, label_list, args, eval_ATE=not args.use_bert_spc)
    mlflow.log_metrics({"accuracy": test_apc_result["apc_test_acc"], "f1_macro": test_apc_result["apc_test_f1_macro"],
                        "f1_micro": test_apc_result["apc_test_f1_micro"], "f1_weighted": test_apc_result["apc_test_f1_weighted"]})

    print('#' * 80)
    print("test apc results: ", test_apc_result)


if __name__ == "__main__":

    with open('experiments.json', "r", encoding='utf-8') as reader:
        config = json.loads(reader.read())

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="laptop", type=str)
    parser.add_argument("--output_dir", default=config['output_dir'], type=str)
    parser.add_argument("--SRD", default=int(config['SRD']), type=int)
    parser.add_argument("--learning_rate", default=float(config['learning_rate']), type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--use_unique_bert", default=bool(config['use_unique_bert']), type=bool,
                        help="Set use_unique_bert = true to use a unique BERT layer to model for both local and global contexts in case of out-of-memory problem")
    parser.add_argument("--use_bert_spc", default=bool(config['use_bert_spc_for_apc']), type=bool,
                        help="Set use_bert_spc = True to improve the APC performance while only APC is considered.")
    parser.add_argument("--local_context_focus", default=config['local_context_focus'], type=str)
    parser.add_argument("--num_train_epochs", default=float(config['num_train_epochs']), type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", default=int(config['train_batch_size']), type=int,
                        help="Total batch size for training.")
    parser.add_argument("--dropout", default=float(config['dropout']), type=int)
    parser.add_argument("--max_seq_length", default=int(config['max_seq_length']), type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Total batch size for eval.")
    parser.add_argument("--eval_steps", default=20, help="evaluate per steps")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--train_seed", default=0, type=int, help="choose train/val split for semeval data")
    parser.add_argument("--seed", default=0, type=int, help="for reproducibility")
    parser.add_argument("--patience", default=float(config['patience']), type=int, help="number of epochs to wait for early stopping after best epoch")
    parser.add_argument("--run_name", default=None, type=str, help="run name for mlflow")

    config = parser.parse_args()

    from utils.Pytorch_GPUManager import GPUManager

    index = GPUManager().auto_choice()
    device = torch.device("cuda:" + str(index) if torch.cuda.is_available() else "cpu")

    config.device = device

    mlflow.set_tracking_uri("../mlruns")
    mlflow.set_experiment("lcf-atepc")
    
    with mlflow.start_run(run_name=config.run_name):
        main(config)