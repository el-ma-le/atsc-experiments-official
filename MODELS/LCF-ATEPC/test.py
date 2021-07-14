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
    id_results = {}
    n_test_correct, n_test_total = 0, 0
    test_apc_logits_all, test_polarities_all = None, None
    model.eval()
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    for batch in dataloader:

        if "arts" in args.dataset:
            input_ids_spc, input_mask, segment_ids, label_ids, polarities, valid_ids, l_mask, ids = batch
        else:
            input_ids_spc, input_mask, segment_ids, label_ids, polarities, valid_ids, l_mask = batch

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

            if "arts" in args.dataset:

                for (id, predicted, lab) in zip(ids, torch.argmax(apc_logits, -1).detach().cpu().numpy(), polarities.detach().cpu().numpy()):

                        id = str(id.detach().cpu().numpy())

                        if id in id_results.keys():
                            id_results[id]["correct"] += int(lab == predicted)
                            id_results[id]["total"] += 1
                        else:
                            id_results[id] = {}
                            id_results[id]["correct"] = int(lab == predicted)
                            id_results[id]["total"] = 1        
        
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

        if len(id_results) != 0:
            # Aspect Robustness Score (ARS)
            id_preds = []
            id_total = len(id_results.keys()) * [1]

            for id in id_results.keys():
                if id_results[id]["correct"] == id_results[id]["total"]:
                    id_preds += [1]
                else:
                    id_preds += [0]

            acc_ars = accuracy_score(id_total, id_preds)

        else:
            acc_ars = -1000


        apc_result = {'apc_test_f1_macro': test_f1_macro, 'apc_test_f1_micro': test_f1_micro, 
                        'apc_test_f1_weighted': test_f1_weighted, "apc_test_acc": test_acc_sk, "apc_test_acc_ars": acc_ars}
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

        if hasattr(features[0], "sentence_id"):
            sentence_ids = torch.tensor([f.sentence_id for f in features], dtype=torch.double)
            new_data = TensorDataset(all_spc_input_ids, all_input_mask, all_segment_ids,
                                   all_label_ids, all_polarities, all_valid_ids, all_lmask_ids, sentence_ids)
        else:
            new_data = TensorDataset(all_spc_input_ids, all_input_mask, all_segment_ids,
                                   all_label_ids, all_polarities, all_valid_ids, all_lmask_ids)

        if eval == False:
            new_sampler = SequentialSampler(new_data)
        else:
            new_sampler = RandomSampler(new_data)

        return DataLoader(new_data, sampler=new_sampler, batch_size=batch_size)


    args = config

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    processor = ATEPCProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1

    datasets = {
        'laptop': "/home/ubuntu/mylrz/atsc-experiments/DATA/SEMEVAL-14-LAP",
        'restaurant': "/home/ubuntu/mylrz/atsc-experiments/DATA/SEMEVAL-14-REST",
        'mams':  "/home/ubuntu/mylrz/atsc-experiments/DATA/MAMS",
        'arts-lap': "/home/ubuntu/mylrz/atsc-experiments/DATA/ARTS-LAP",
        'arts-rest': "/home/ubuntu/mylrz/atsc-experiments/DATA/ARTS-REST"
    }
    pretrained_bert_models = {
        'laptop': "bert-base-uncased",
        'restaurant': "bert-base-uncased",
        # for loading domain-adapted BERT
        # 'restaurant': "../bert_pretrained_restaurant",
        'mams': "bert-base-uncased",
        'arts-lap': "bert-base-uncased",
        'arts-rest': "bert-base-uncased",
    }

    args.bert_model = pretrained_bert_models[args.dataset]
    args.data_dir = datasets[args.dataset]

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    bert_base_model = BertModel.from_pretrained(args.bert_model)
    bert_base_model.config.num_labels = num_labels

    for arg in vars(args):
        print('>>> {0}: {1}'.format(arg, getattr(args, arg)))
        mlflow.log_param(arg, getattr(args, arg))

   
    # Evaluate on test data
    test_examples = processor.get_test_examples(args.data_dir)
    test_dataloader = create_dataloader(test_examples, label_list, args.max_seq_length, tokenizer, args.eval_batch_size)
    
    model = mlflow.pytorch.load_model(args.model)
    model.to(device)

    test_apc_result, test_ate_result = evaluate(test_dataloader, model, label_list, args, eval_ATE=not args.use_bert_spc)
    mlflow.log_metrics({"accuracy": test_apc_result["apc_test_acc"], "f1_macro": test_apc_result["apc_test_f1_macro"],
                        "f1_micro": test_apc_result["apc_test_f1_micro"], "f1_weighted": test_apc_result["apc_test_f1_weighted"],
                        "accuracy_ars": test_apc_result["apc_test_acc_ars"]})

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
    parser.add_argument("--dropout", default=float(config['dropout']), type=int)
    parser.add_argument("--max_seq_length", default=int(config['max_seq_length']), type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Total batch size for eval.")
    parser.add_argument("--eval_steps", default=20, help="evaluate per steps")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--run_name", default=None, type=str, help="run name for mlflow")
    parser.add_argument("--model", default=None, type=str, help="model to be used for testing")

    config = parser.parse_args()

    from utils.Pytorch_GPUManager import GPUManager

    index = GPUManager().auto_choice()
    device = torch.device("cuda:" + str(index) if torch.cuda.is_available() else "cpu")

    config.device = device

    mlflow.set_tracking_uri("/home/ubuntu/mylrz/atsc-experiments/MODELS/mlruns")
    mlflow.set_experiment("lcf-atepc")
    
    with mlflow.start_run(run_name=config.run_name):
        main(config)