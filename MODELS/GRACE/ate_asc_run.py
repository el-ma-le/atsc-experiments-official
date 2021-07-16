from __future__ import absolute_import, division, print_function
from ate_run import DATASET_DICT

import argparse
#import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from ate_asc_modeling import BertForSequenceLabeling
from optimization import BertAdam
from tokenization import BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from ate_asc_features import ATEASCProcessor, convert_examples_to_features, get_labels
from utils import get_logger, get_aspect_chunks, get_polaity_chunks

import mlflow
import copy

SMALL_POSITIVE_CONST = 1e-4

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return max(0, (1.0 - x) / (1.0 - warmup))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def parse_input_parameter():
    #global logger
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, bert-large-uncased.")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--task_name", default="ate_asc", type=str, required=False, help="The name of the task to train.")
    parser.add_argument("--data_name", default="", type=str, required=False, help="The name of the task to train.")
    parser.add_argument("--train_file", default=None, type=str, required=False)
    parser.add_argument("--valid_file", default=None, type=str, required=False)
    parser.add_argument("--test_file", default=None, type=str, required=False)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \nSequences longer than this will be truncated, and sequences shorter \nthan this will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--num_thread_reader', type=int, default=0, help='')
    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=1, help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n0 (default value): dynamic loss scaling.\nPositive power of 2: static loss scaling value.\n")
    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. A number of warnings are expected for a normal CoQA evaluation.")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.");
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

    parser.add_argument("--use_ghl", action='store_true', help="Whether use weighted cross entropy to decoder.")
    parser.add_argument("--use_vat", action='store_true', help="Whether use vat to encoder.")

    parser.add_argument("--use_decoder", default=True, type=str2bool, help="Whether use decoder to asc.")
    parser.add_argument("--num_decoder_layer", default=2, type=int, help="When `use_decoder' is True, set the number of decoder.")
    parser.add_argument("--decoder_shared_layer", default=3, type=int, help="When `use_decoder' is True, set the number of shared encoder.")

    parser.add_argument("--patience", type=int, default=1, help="number of epochs after best one to wait until early stopping")

    args = parser.parse_args()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    #logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    print("Effective parameters:")
    for key in sorted(args.__dict__):
        print("  {}: {}".format(key, args.__dict__[key]))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    task_config = {
        "use_ghl": args.use_ghl,
        "use_vat": args.use_vat,
        "num_decoder_layer": args.num_decoder_layer,
        "decoder_shared_layer": args.decoder_shared_layer,
    }

    return args, task_config

def init_device(args):
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    return device, n_gpu

def init_model(args, num_tp_labels, task_config, device, n_gpu):

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
        if args.num_decoder_layer != 0:
            layer_num_list = [layer_num for layer_num in range(args.num_decoder_layer)]
            new_model_state_dict = {}
            model_state_dict_exsit_keys = model_state_dict.keys()
            max_bert_layer = max([int(k_str.split(".")[3]) for k_str in model_state_dict_exsit_keys if "bert.encoder.layer" in k_str])
            for k_str in model_state_dict_exsit_keys:
                new_model_state_dict[k_str] = model_state_dict[k_str]
                for layer_num in layer_num_list:
                    bert_key_name = "bert.encoder.layer.{}".format(max_bert_layer - args.num_decoder_layer + 1 + layer_num)
                    mirror_key_name = "bert.encoder.layer.{}".format(layer_num)
                    if k_str.find(bert_key_name) == 0:
                        new_key_name = k_str.replace(bert_key_name, mirror_key_name).replace("bert.encoder", "decoder.decoder")
                        if "attention.self" in new_key_name:
                            new_key_name_sufx = new_key_name.replace("attention.self", "slf_attn.att")
                            new_model_state_dict[new_key_name_sufx] = model_state_dict[k_str].clone()
                            new_key_name_sufx = new_key_name.replace("attention.self", "enc_attn.att")
                            new_model_state_dict[new_key_name_sufx] = model_state_dict[k_str].clone()
                        elif "attention.output" in new_key_name:
                            new_key_name_sufx = new_key_name.replace("attention.output", "slf_attn.output")
                            new_model_state_dict[new_key_name_sufx] = model_state_dict[k_str].clone()
                            new_key_name_sufx = new_key_name.replace("attention.output", "enc_attn.output")
                            new_model_state_dict[new_key_name_sufx] = model_state_dict[k_str].clone()
                        else:
                            new_model_state_dict[new_key_name] = model_state_dict[k_str].clone()

                if k_str.find("bert.embeddings") == 0:
                    new_key_name = k_str.replace("bert.embeddings", "decoder.embeddings")
                    new_model_state_dict[new_key_name] = model_state_dict[k_str].clone()

            model_state_dict = new_model_state_dict
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else \
        os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    model = BertForSequenceLabeling.from_pretrained(args.bert_model, cache_dir=cache_dir, state_dict=model_state_dict,
                                                    num_tp_labels=num_tp_labels, task_config=task_config)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    return model

def prep_optimizer(args, model, num_train_optimization_steps):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    if args.use_decoder:
        no_decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
        decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

        no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." in n]
        no_decay_nobert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." not in n]

        decay_bert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." in n]
        decay_nobert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." not in n]

        coef_lr = 10.
        optimizer_grouped_parameters = [
            {'params': [p for n, p in no_decay_bert_param_tp], 'weight_decay': 0.01},
            {'params': [p for n, p in no_decay_nobert_param_tp], 'weight_decay': 0.01, 'lr': args.learning_rate * coef_lr},
            {'params': [p for n, p in decay_bert_param_tp], 'weight_decay': 0.0},
            {'params': [p for n, p in decay_nobert_param_tp], 'weight_decay': 0.0, 'lr': args.learning_rate * coef_lr}
        ]
    else:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters, lr=args.learning_rate, bias_correction=False, max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
    return optimizer

def dataloader_train(args, tokenizer, file_path):
    dataset = ATEASCProcessor(file_path=file_path, set_type="train")
    print("Loaded train file: {}".format(file_path))
    # get label lists
    at_labels, as_labels = get_labels(dataset.label_tp_list)

    features = convert_examples_to_features(dataset.examples, (at_labels, as_labels),
                                            args.max_seq_length, tokenizer, verbose_logging=args.verbose_logging)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_at_label_ids = torch.tensor([f.at_label_id for f in features], dtype=torch.long)
    all_as_label_ids = torch.tensor([f.as_label_id for f in features], dtype=torch.long)

    all_label_mask = torch.tensor([f.label_mask for f in features], dtype=torch.long)
    all_label_mask_X = torch.tensor([f.label_mask_X for f in features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_at_label_ids, all_as_label_ids,
                               all_label_mask, all_label_mask_X)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=args.num_thread_reader)

    return dataloader, train_data, (at_labels, as_labels)

def dataloader_val(args, tokenizer, file_path, label_tp_list, set_type="val"):

    dataset = ATEASCProcessor(file_path=file_path, set_type=set_type)
    print("Loaded val file: {}".format(file_path))

    eval_features = convert_examples_to_features(dataset.examples, label_tp_list,
                                                 args.max_seq_length, tokenizer, verbose_logging=args.verbose_logging)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_at_label_ids = torch.tensor([f.at_label_id for f in eval_features], dtype=torch.long)
    all_as_label_ids = torch.tensor([f.as_label_id for f in eval_features], dtype=torch.long)

    all_label_mask = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
    all_label_mask_X = torch.tensor([f.label_mask_X for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_at_label_ids, all_as_label_ids,
                              all_label_mask, all_label_mask_X)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    return eval_dataloader, eval_data

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, tokenizer, optimizer, global_step, num_train_optimization_steps):
    #global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = 100
    start_time = time.time()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    weight_gradient = None  # Init in model: [bin_num]
    weight_gradient_labels = None  # Init in model:
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, at_label_ids, as_label_ids, label_mask, label_mask_X = batch
        loss, acc_sum, weight_gradient, weight_gradient_labels = model(input_ids, segment_ids, input_mask, label_mask_X,
                                                                       at_label_ids, as_label_ids,
                                                                       weight_gradient, weight_gradient_labels)
        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        tr_loss += float(loss.item())
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                # modify learning rate with special warm up BERT uses
                # if args.fp16 is False, BertAdam is used that handles this automatically
                lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % log_step == 0:
                print("Epoch: {}/{}, Step: {}/{}, Lr: {}, Loss: {}, Time/step: {}".format(epoch + 1,
                            args.num_train_epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.6f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss.item()), (time.time() - start_time) / (log_step * args.gradient_accumulation_steps)))
                start_time = time.time()

    tr_loss = tr_loss / len(train_dataloader)
    return tr_loss, global_step

def cal_f1_at(y_true, y_pred):
    correct_pred, total_ground, total_pred = 0., 0., 0.
    lab_chunks_list, lab_pred_chunks_list = [], []

    for ground_seq, pred_seq in zip(y_true, y_pred):
        # groudn_seq, pred_seq: sequence of labels, e.g. [B-AP, O, B-AP, B-AP, I-AP, O, B-AP, I-AP, I-AP, O, B-AP]
        # lab_chunks, lab_pred_chunks: list of (type, beg, end), e.g. [('AP', 0, 1), ('AP', 2, 3), ('AP', 3, 5), ('AP', 6, 9), ('AP', 10, 11)]
        lab_chunks = get_aspect_chunks(ground_seq, default="O")
        lab_pred_chunks = get_aspect_chunks(pred_seq, default="O")
        lab_chunks_list.append(lab_chunks)
        lab_pred_chunks_list.append(lab_pred_chunks)

        # remove duplicate triples
        lab_chunks = set(lab_chunks)
        lab_pred_chunks = set(lab_pred_chunks)

        # aspect detection is correct if triples are identical in true and predicted
        correct_pred += len(lab_chunks & lab_pred_chunks)
        total_pred += len(lab_pred_chunks)
        total_ground += len(lab_chunks)

    p = correct_pred / total_pred if total_pred > 0 else 0.
    r = correct_pred / total_ground if total_ground > 0 else 0.
    f1_micro = 2 * p * r / (p + r) if p > 0 and r > 0 else 0.
    return p, r, f1_micro, lab_chunks_list, lab_pred_chunks_list

def cal_f1_as(y_true, y_pred, at_lab_chunks_list, at_lab_pred_chunks_list, must_predict=False):
    correct_pred, total_ground, total_pred = 0., 0., 0.
    lab_chunks_list, lab_pred_chunks_list = [], []
    n_total_hits, n_total_gold, n_total_pred = 0, 0, 0

    for ground_seq, pred_seq, lab_chunks, lab_pred_chunks in zip(y_true, y_pred, at_lab_chunks_list, at_lab_pred_chunks_list):

        # lab_chunkgs, lab_pred_chunks: list of (pol, beg, end), e.g.  [('POSITIVE', 0, 1), ('POSITIVE', 2, 3), ('POSITIVE', 3, 5), ('NEUTRAL', 6, 9), ('POSITIVE', 10, 11)]
        lab_chunks = get_polaity_chunks(ground_seq, lab_chunks, default="O", must_predict=must_predict)
        lab_pred_chunks = get_polaity_chunks(pred_seq, lab_pred_chunks, default="O", must_predict=must_predict)
        lab_chunks_list.append(lab_chunks)
        lab_pred_chunks_list.append(lab_pred_chunks)

        # remove duplicate triples
        lab_chunks = set(lab_chunks)
        lab_pred_chunks = set(lab_pred_chunks)

        # sentiment classification is correct if triples are identical in true and predicted
        # overall counters
        correct_pred += len(lab_chunks & lab_pred_chunks)
        total_pred += len(lab_pred_chunks)
        total_ground += len(lab_chunks)

        # polarity-specific counters (for macro f1)
        # taken from BERT+ code
        tag2tagid = {'POSITIVE': 0, 'NEGATIVE': 1, 'NEUTRAL': 2}
        hit_count, gold_count, pred_count = np.zeros(3), np.zeros(3), np.zeros(3)
        for triple in lab_chunks:
            pol_tag = triple[0]
            tid = tag2tagid[pol_tag]
            gold_count[tid] += 1

        for triple in lab_pred_chunks:
            pol_tag = triple[0]
            tid = tag2tagid[pol_tag]
            if triple in lab_chunks:
                hit_count[tid] += 1
            pred_count[tid] += 1

        n_total_hits += hit_count
        n_total_gold += gold_count
        n_total_pred += pred_count

    assert (sum(n_total_hits) == correct_pred), "not identical number of hits"
    assert (sum(n_total_gold) == total_ground), "not identical number of gold"
    assert (sum(n_total_pred) == total_pred), "not identical number of preds"

    ts_precision, ts_recall, ts_f1 = np.zeros(3), np.zeros(3), np.zeros(3)
    for i in range(3):

        n_ts = n_total_hits[i]
        n_g_ts = n_total_gold[i]
        n_p_ts = n_total_pred[i]

        ts_precision[i] = float(n_ts) / float(n_p_ts + SMALL_POSITIVE_CONST)
        ts_recall[i] = float(n_ts) / float(n_g_ts + SMALL_POSITIVE_CONST)
        
        ts_f1[i] = 2 * ts_precision[i] * ts_recall[i] / (ts_precision[i] + ts_recall[i] + SMALL_POSITIVE_CONST)

    f1_macro = ts_f1.mean()

    p = correct_pred / (total_pred + SMALL_POSITIVE_CONST) if total_pred > 0 else 0.
    r = correct_pred / (total_ground + SMALL_POSITIVE_CONST) if total_ground > 0 else 0.
    f1_micro = 2 * p * r / (p + r + SMALL_POSITIVE_CONST) if p > 0 and r > 0 else 0.
    return p, r, f1_micro, f1_macro, lab_chunks_list, lab_pred_chunks_list

def eval_epoch(model, eval_dataloader, label_tp_list, device):
    if hasattr(model, 'module'):
        model = model.module
    model.eval()

    y_true_at = []
    y_pred_at = []
    y_true_as = []
    y_pred_as = []
    at_label_list, as_label_list = label_tp_list
    at_label_map = {i: label for i, label in enumerate(at_label_list)}
    as_label_map = {i: label for i, label in enumerate(as_label_list)}
    for input_ids, input_mask, segment_ids, at_label_ids, as_label_ids, label_mask, label_mask_X in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        at_label_ids = at_label_ids.to(device)
        as_label_ids = as_label_ids.to(device)
        label_mask = label_mask.to(device)
        label_mask_X = label_mask_X.to(device)

        with torch.no_grad():
            # logits, decoder_logits = model(input_ids, segment_ids, input_mask)
            logits, sequence_output, encoder_output = model.get_encoder_logits(input_ids, segment_ids, input_mask)
            pred_dec_ids = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
            decoder_logits = model.get_decoder_logits(encoder_output, input_mask, label_mask_X, pred_dec_ids)
            logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
            decoder_logits = torch.argmax(F.log_softmax(decoder_logits, dim=2), dim=2)
            logits = logits.detach().cpu().numpy()
            decoder_logits = decoder_logits.detach().cpu().numpy()

        at_label_ids = at_label_ids.to('cpu').numpy()
        as_label_ids = as_label_ids.to('cpu').numpy()
        label_mask = label_mask.to('cpu').numpy()

        # for each review i, get mask_i
        for i, mask_i in enumerate(label_mask):
            temp_11 = []
            temp_12 = []
            temp_21 = []
            temp_22 = []
            # for each token, get the mask value l at position j
            for j, l in enumerate(mask_i):
                # if mask value l does not indicate "no aspect",
                # add true and predicted labels for aspects and pols to lists
                if l > -1:
                    temp_11.append(at_label_map[at_label_ids[i][j]])
                    temp_12.append(at_label_map[logits[i][j]])
                    temp_21.append(as_label_map[as_label_ids[i][j]])
                    temp_22.append(as_label_map[decoder_logits[i][j]])

            y_true_at.append(temp_11)
            y_pred_at.append(temp_12)
            y_true_as.append(temp_21)
            y_pred_as.append(temp_22)

    # calculate measures based on triples
    p, r, at_f1_micro, at_lab_chunks_list, at_lab_pred_chunks_list = cal_f1_at(y_true_at, y_pred_at)
    print("AT p: {:.4f}\tr: {:.4f}\tf1_micro: {:.4f}".format(p, r, at_f1_micro))

    # calculate measures based on triples
    as_lab_p, as_lab_r, as_lab_f1_micro, as_lab_f1_macro, _, _ = cal_f1_as(y_true_as, y_pred_as, at_lab_chunks_list, at_lab_pred_chunks_list)
    print("AS p: {:.4f}\tr: {:.4f}\tf1_micro: {:.4f}\tf1_macro: {:.4f}".format(as_lab_p, as_lab_r, as_lab_f1_micro, as_lab_f1_macro))

    return as_lab_p, as_lab_r, as_lab_f1_micro, as_lab_f1_macro, at_f1_micro

def save_model(epoch, args, model):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}".format(epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    print("Model saved to ", output_model_file)
    return output_model_file

def load_model(epoch, args, num_tp_labels, task_config, device):
    model_file = os.path.join(
        args.output_dir,
        "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        print("Model loaded from ", model_file)
        model = BertForSequenceLabeling.from_pretrained(args.bert_model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
                                                        state_dict=model_state_dict, num_tp_labels=num_tp_labels,
                                                        task_config=task_config)
        model.to(device)
    else:
        model = None
    return model


DATALOADER_DICT = {}
DATALOADER_DICT["ate_asc"] = {"train":dataloader_train, "eval":dataloader_val}

DATASET_DICT = {}
for i in range(1,6):
    DATASET_DICT["lap{}".format(i)] = {"train_file":"SEMEVAL-14-LAP/grace_train_{}.txt".format(i), 
                        "valid_file":"SEMEVAL-14-LAP/grace_val_{}.txt".format(i), 
                        "test_file":"SEMEVAL-14-LAP/grace_test.txt"}
    DATASET_DICT["rest{}".format(i)] = {"train_file":"SEMEVAL-14-REST/grace_train_{}.txt".format(i), 
                        "valid_file":"SEMEVAL-14-REST/grace_val_{}.txt".format(i), 
                        "test_file":"SEMEVAL-14-REST/grace_test.txt"}
DATASET_DICT["mams"] = {"train_file": "MAMS/grace_train.txt",
                        "valid_file": "MAMS/grace_val.txt",
                        "test_file": "MAMS/grace_test.txt"}


def main():
    #global logger

    mlflow.set_tracking_uri("/home/ubuntu/mylrz/atsc-experiments/MODELS/mlruns")
    mlflow.set_experiment("grace")

    mlflow.start_run()

    args, task_config = parse_input_parameter()

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device, n_gpu = init_device(args)

    data_name = args.data_name.lower()
    if data_name in DATASET_DICT:
        args.train_file = DATASET_DICT[data_name]["train_file"]
        args.valid_file = DATASET_DICT[data_name]["valid_file"]
        args.test_file = DATASET_DICT[data_name]["test_file"]
    else:
        assert args.train_file is not None
        assert args.valid_file is not None
        assert args.test_file is not None

    # log arguments
    for arg in vars(args):
        print('>>> {0}: {1}'.format(arg, getattr(args, arg)))
        mlflow.log_param(arg, getattr(args, arg))

    task_name = args.task_name.lower()
    if task_name not in DATALOADER_DICT:
        raise ValueError("Task not found: %s " % (task_name))

    if n_gpu > 1 and (args.use_ghl):
        print("Multi-GPU make the results not reproduce.")

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Generate label list from training dataset
    file_path = os.path.join(args.data_dir, args.train_file)
    train_dataloader, train_examples, label_tp_list = DATALOADER_DICT[task_name]["train"](args, tokenizer, file_path)

    print("AT Labels are =  :", "["+", ".join(label_tp_list[0])+"]")
    print("AS Labels are =  :", "["+", ".join(label_tp_list[1])+"]")
    at_num_labels = len(label_tp_list[0])
    as_num_labels = len(label_tp_list[1])
    num_tp_labels = (at_num_labels, as_num_labels)

    task_config["at_labels"] = label_tp_list[0]
    model = init_model(args, num_tp_labels, task_config, device, n_gpu)

    # Generate test dataset
    file_path = os.path.join(args.data_dir, args.test_file)
    test_dataloader, test_examples = DATALOADER_DICT[task_name]["eval"](args, tokenizer, file_path,
                                                                        label_tp_list=label_tp_list, set_type="test")

    if args.do_train:
        num_train_optimization_steps = (int(len(
            train_dataloader) + args.gradient_accumulation_steps - 1) / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        optimizer = prep_optimizer(args, model, num_train_optimization_steps)

        print("***** Running training *****")
        print("  Num examples = ", len(train_examples))
        print("  Batch size = ", args.train_batch_size)
        print("  Num steps = ", num_train_optimization_steps)

        file_path = os.path.join(args.data_dir, args.valid_file)
        eval_dataloader, eval_examples = DATALOADER_DICT[task_name]["eval"](args, tokenizer, file_path,
                                                                            label_tp_list=label_tp_list, set_type="val")
        print("***** Running evaluation *****")
        print("  Num examples = ", len(eval_examples))
        print("  Batch size = ", args.eval_batch_size)

        global_step = 0
        max_val_epoch = 0
        max_as_f1_macro = 0
        max_as_f1_micro = 0
        max_at_f1_micro = 0

        for epoch in range(args.num_train_epochs):
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, tokenizer,
                                               optimizer, global_step, num_train_optimization_steps)
            print("Epoch {}/{}  Finished, Train Loss: {}".format(epoch + 1, args.num_train_epochs, tr_loss))
            #save_model(epoch, args, model)
            _, _, val_as_f1_micro, val_as_f1_macro, val_at_f1_micro = eval_epoch(model, eval_dataloader, label_tp_list, device)
            mlflow.log_metrics({"val_as_f1_micro": val_as_f1_micro, "val_as_f1_macro": val_as_f1_macro, "val_at_f1_micro": val_at_f1_micro}, epoch)

            if val_as_f1_micro >= max_as_f1_micro:
                max_as_f1_micro = val_as_f1_micro
                max_val_epoch = epoch
                best_model = copy.deepcopy(model)

            if val_as_f1_macro >= max_as_f1_macro:
                max_as_f1_macro = val_as_f1_macro
                max_val_epoch = epoch
                best_model = copy.deepcopy(model)

            if val_at_f1_micro >= max_at_f1_micro:
                max_at_f1_micro = val_at_f1_micro
                max_val_epoch = epoch
                best_model = copy.deepcopy(model)

            if epoch - max_val_epoch >= args.patience:
                print('>> early stop.')
                mlflow.log_metric("early_stopp_at_epoch", epoch)
                break

        mlflow.pytorch.log_model(best_model, "model")

        print("***** Running test *****")
        print("  Num examples = ", len(test_examples))
        print("  Batch size = ", args.eval_batch_size)
        print("***Results on test***")

        test_as_precision, test_as_recall, test_as_f1_micro, test_as_f1_macro, test_at_f1_micro = eval_epoch(best_model, test_dataloader, label_tp_list, device)
        mlflow.log_metrics({"as_f1_micro": test_as_f1_micro, "as_f1_macro": test_as_f1_macro, "at_f1_micro": test_at_f1_micro, 
                            "as_precision": test_as_precision, "as_recall": test_as_recall})

    elif args.do_eval:

        print("***** Running test *****")
        print("  Num examples = ", len(test_examples))
        print("  Batch size = ", args.eval_batch_size)
        print("***Results on test***")

        test_as_precision, test_as_recall, test_as_f1_micro, test_as_f1_macro, test_at_f1_micro = eval_epoch(model, test_dataloader, label_tp_list, device)
        mlflow.log_metrics({"as_f1_micro": test_as_f1_micro, "as_f1_macro": test_as_f1_macro, "at_f1_micro": test_at_f1_micro, 
                            "as_precision": test_as_precision, "as_recall": test_as_recall})
    else:
        if args.init_model:

            print("***** Running test *****")
            print("  Num examples = ", len(test_examples))
            print("  Batch size = ", args.eval_batch_size)
            print("***Results on test***")

            test_as_precision, test_as_recall, test_as_f1_micro, test_as_f1_macro, test_at_f1_micro = eval_epoch(model, test_dataloader, label_tp_list, device)
            mlflow.log_metrics({"as_f1_micro": test_as_f1_micro, "as_f1_macro": test_as_f1_macro, "at_f1_micro": test_at_f1_micro, 
                            "as_precision": test_as_precision, "as_recall": test_as_recall})

        else:
            for epoch in range(args.num_train_epochs):
                # Load a trained model that you have fine-tuned
                model = load_model(epoch, args, num_tp_labels, task_config, device)
                if not model:
                    break
                print("***** Running test *****")
                print("  Num examples = ", len(test_examples))
                print("  Batch size = ", args.eval_batch_size)
                print("***Results on test***")

                test_as_precision, test_as_recall, test_as_f1_micro, test_as_f1_macro, test_at_f1_micro = eval_epoch(model, test_dataloader, label_tp_list, device)
                mlflow.log_metrics({"as_f1_micro": test_as_f1_micro, "as_f1_macro": test_as_f1_macro, "at_f1_micro": test_at_f1_micro, 
                            "as_precision": test_as_precision, "as_recall": test_as_recall})

    mlflow.end_run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard break~")