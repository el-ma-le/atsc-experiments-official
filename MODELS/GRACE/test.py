from __future__ import absolute_import, division, print_function
from ate_run import DATASET_DICT

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from ate_asc_modeling import BertForSequenceLabeling
from optimization import BertAdam
from tokenization import BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from ate_asc_features import ATEASCProcessor, ATEASCProcessor_arts, convert_examples_to_features, get_labels
from utils import get_logger, get_aspect_chunks, get_polaity_chunks

from sklearn import metrics
from tqdm import tqdm
import mlflow

SMALL_POSITIVE_CONST = 1e-4

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def parse_input_parameter():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, bert-large-uncased.")
    #parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--task_name", default="ate_asc", type=str, required=False, help="The name of the task to train.")
    parser.add_argument("--data_name", default="", type=str, required=False, help="The name of the task to train.")
    parser.add_argument("--train_file", default=None, type=str, required=False)
    parser.add_argument("--valid_file", default=None, type=str, required=False)
    parser.add_argument("--test_file", default=None, type=str, required=False)
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \nSequences longer than this will be truncated, and sequences shorter \nthan this will be padded.")
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

    parser.add_argument("--run_name", type=str, default=None, help="run name for mlflow")
    parser.add_argument("--logged_model", default=None, help="model to be used for testing")
    parser.add_argument("--split", default=0, type=int, help="train/val split in semeval data")

    args = parser.parse_args()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    #if not os.path.exists(args.output_dir):
    #    os.makedirs(args.output_dir)

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


def labelloader_train(file_path):
    dataset = ATEASCProcessor(file_path=file_path, set_type="train")
    print("Loaded train file: {}".format(file_path))
    # get label lists
    at_labels, as_labels = get_labels(dataset.label_tp_list)

    return (at_labels, as_labels)


def dataloader_val(args, tokenizer, file_path, label_tp_list, set_type="val"):

    if "arts" in args.data_name.lower():
        dataset = ATEASCProcessor_arts(file_path=file_path, set_type=set_type)
    else:
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

    if hasattr(eval_features[0], "sentence_ids"):
        sentence_ids = torch.tensor([f.sentence_ids for f in eval_features], dtype=torch.double)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_at_label_ids, all_as_label_ids,
                              all_label_mask, all_label_mask_X, sentence_ids)
    else:
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_at_label_ids, all_as_label_ids,
                              all_label_mask, all_label_mask_X)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    return eval_dataloader, eval_data


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

def cal_f1_as_arts(y_true, y_pred, at_lab_chunks_list, at_lab_pred_chunks_list, sentence_ids, must_predict=False):
    correct_pred, total_ground, total_pred = 0., 0., 0.
    lab_chunks_list, lab_pred_chunks_list = [], []
    n_total_hits, n_total_gold, n_total_pred = 0, 0, 0
    id_results = {}

    for ground_seq, pred_seq, lab_chunks, lab_pred_chunks, sentence_id in zip(y_true, y_pred, at_lab_chunks_list, at_lab_pred_chunks_list, sentence_ids):

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

        # for ARS, assign hits and totals to ids
        if sentence_id in id_results.keys():
            id_results[sentence_id]["correct"] += len(lab_chunks & lab_pred_chunks)
            id_results[sentence_id]["total"] += len(lab_chunks)
        else:
            id_results[sentence_id] = {}
            id_results[sentence_id]["correct"] = len(lab_chunks & lab_pred_chunks)
            id_results[sentence_id]["total"] = len(lab_chunks)

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
    
    # Aspect Robustness Score (ARS)
    id_preds = []
    id_total = len(id_results.keys()) * [1]

    for id in id_results.keys():
        if id_results[id]["correct"] == id_results[id]["total"]:
            id_preds += [1]
        else:
            id_preds += [0]
    
    acc_ars = metrics.accuracy_score(id_total, id_preds)
    
    return p, r, f1_micro, f1_macro, acc_ars, lab_chunks_list, lab_pred_chunks_list

def eval_epoch(model, eval_dataloader, label_tp_list, device, data_name):
    if hasattr(model, 'module'):
        model = model.module
    model.eval()

    y_true_at = []
    y_pred_at = []
    y_true_as = []
    y_pred_as = []
    all_ids = []
    at_label_list, as_label_list = label_tp_list
    at_label_map = {i: label for i, label in enumerate(at_label_list)}
    as_label_map = {i: label for i, label in enumerate(as_label_list)}

    for batch in tqdm(eval_dataloader, desc="Evaluating"):

        batch = tuple(t.to(device) for t in batch)

        if "arts" in data_name: 
            input_ids, input_mask, segment_ids, at_label_ids, as_label_ids, label_mask, label_mask_X, ids = batch
            all_ids.append(ids)
        else:
            input_ids, input_mask, segment_ids, at_label_ids, as_label_ids, label_mask, label_mask_X = batch

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        at_label_ids = at_label_ids.to(device)
        as_label_ids = as_label_ids.to(device)
        label_mask = label_mask.to(device)
        label_mask_X = label_mask_X.to(device)

        with torch.no_grad():
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

    # calculate aspect detection measures based on triples
    p, r, at_f1_micro, at_lab_chunks_list, at_lab_pred_chunks_list = cal_f1_at(y_true_at, y_pred_at)
    print("AT p: {:.4f}\tr: {:.4f}\tf1_micro: {:.4f}".format(p, r, at_f1_micro))

    if len(all_ids) == 0:
        # calculate polarity measures based on triples
        as_lab_p, as_lab_r, as_lab_f1_micro, as_lab_f1_macro, _, _ = cal_f1_as(y_true_as, y_pred_as, at_lab_chunks_list, at_lab_pred_chunks_list)
        print("AS p: {:.4f}\tr: {:.4f}\tf1_micro: {:.4f}\tf1_macro: {:.4f}".format(as_lab_p, as_lab_r, as_lab_f1_micro, as_lab_f1_macro))

        return as_lab_p, as_lab_r, as_lab_f1_micro, as_lab_f1_macro, at_f1_micro, -1000

    else:
        # calculate polarity measures based on triples
        as_lab_p, as_lab_r, as_lab_f1_micro, as_lab_f1_macro, as_acc_ars, _, _ = cal_f1_as_arts(y_true_as, y_pred_as, at_lab_chunks_list, at_lab_pred_chunks_list, all_ids)
        print("AS p: {:.4f}\tr: {:.4f}\tf1_micro: {:.4f}\tf1_macro: {:.4f}\tacc_ars: {:.4f}".format(as_lab_p, as_lab_r, as_lab_f1_micro, as_lab_f1_macro, as_acc_ars))

        return as_lab_p, as_lab_r, as_lab_f1_micro, as_lab_f1_macro, at_f1_micro, as_acc_ars


def main():

    mlflow.set_tracking_uri("../mlruns")
    mlflow.set_experiment("grace")

    args, _ = parse_input_parameter()

    mlflow.start_run(run_name=args.run_name)

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device, n_gpu = init_device(args)

    DATASET_DICT = {}
    DATASET_DICT["arts-lap"] = {"test_file": "ARTS-LAP/grace_test.txt",
                                "train_file": "SEMEVAL-14-LAP/grace_train_"+str(args.split)+".txt"}
    DATASET_DICT["arts-rest"] = {"test_file": "ARTS-REST/grace_test.txt",
                                "train_file": "SEMEVAL-14-REST/grace_train_"+str(args.split)+".txt"}
    DATASET_DICT["lap"] = {"test_file": "SEMEVAL-14-LAP/grace_test.txt",
                            "train_file": "SEMEVAL-14-LAP/grace_train_"+str(args.split)+".txt",}

    data_name = args.data_name.lower()
    if data_name in DATASET_DICT:
        args.test_file = DATASET_DICT[data_name]["test_file"]
    else:
        assert args.test_file is not None

    args.train_file = DATASET_DICT[data_name]["train_file"]

    # log arguments
    for arg in vars(args):
        print('>>> {0}: {1}'.format(arg, getattr(args, arg)))
        mlflow.log_param(arg, getattr(args, arg))

    if n_gpu > 1 and (args.use_ghl):
        print("Multi-GPU make the results not reproduce.")

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Generate label list from training dataset
    file_path = os.path.join(args.data_dir, args.train_file)
    label_tp_list = labelloader_train(file_path)

    # load model
    model =  mlflow.pytorch.load_model(args.logged_model)
    model.to(device)
    mlflow.log_param("logged_model", args.logged_model)

    # Generate test dataset
    file_path = os.path.join(args.data_dir, args.test_file)
    test_dataloader, test_examples = dataloader_val(args, tokenizer, file_path,
                                                    label_tp_list=label_tp_list, set_type="test")

    if args.do_eval:

        print("***** Running test *****")
        print("  Num examples = ", len(test_examples))
        print("  Batch size = ", args.eval_batch_size)
        print("***Results on test***")

        test_as_precision, test_as_recall, test_as_f1_micro, test_as_f1_macro, test_at_f1_micro, test_as_acc_ars = eval_epoch(model, test_dataloader, label_tp_list, device, data_name)
        mlflow.log_metrics({"as_f1_micro": test_as_f1_micro, "as_f1_macro": test_as_f1_macro, "at_f1_micro": test_at_f1_micro, 
                            "as_precision": test_as_precision, "as_recall": test_as_recall, "as_acc_ars": test_as_acc_ars})

    mlflow.end_run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard break~")