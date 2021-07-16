# encoding=utf-8
import sys
sys.path.append('../')
import torch
import random
import argparse
import numpy as np
from vocab import Vocab
from utils import helper

from sklearn import metrics
from bert_loader import ABSADataLoader
from bert_trainer import ABSATrainer

import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="SEMEVAL-14-REST")
parser.add_argument("--vocab_dir", type=str, default="SEMEVAL-14-REST")
parser.add_argument("--hidden_dim", type=int, default=768, help="bert dim.")

parser.add_argument("--dep_dim", type=int, default=30, help="dep embedding dimension.")
parser.add_argument("--pos_dim", type=int, default=0, help="pos embedding dimension.")
parser.add_argument("--post_dim", type=int, default=0, help="position embedding dimension.")
parser.add_argument("--num_class", type=int, default=3, help="Num of sentiment class.")

parser.add_argument("--input_dropout", type=float, default=0.1, help="Input dropout rate.")
parser.add_argument("--layer_dropout", type=float, default=0, help="RGAT layer dropout rate.")
parser.add_argument("--att_dropout", type=float, default=0, help="self-attention layer dropout rate.")
parser.add_argument("--lower", default=True, help="Lowercase all words.")
parser.add_argument("--direct", default=False)
parser.add_argument("--loop", default=True)
parser.add_argument("--reset_pooling", default=False, action="store_true")
parser.add_argument("--lr", type=float, default=2e-5, help="learning rate.")
parser.add_argument("--bert_lr", type=float, default=2e-5, help="learning rate for bert.")
parser.add_argument("--l2", type=float, default=1e-5, help="weight decay rate.")
parser.add_argument("--optim",
    choices=["sgd", "adagrad", "adam", "adamax"],
    default="adam",
    help="Optimizer: sgd, adagrad, adam or adamax.",)
parser.add_argument("--num_layer", type=int, default=3, help="Number of graph layers.")
parser.add_argument("--num_epoch", type=int, default=20, help="Number of total training epochs.")
parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
parser.add_argument("--log_step", type=int, default=16, help="Print log every k steps.")
parser.add_argument("--save_dir", type=str, default="./saved_models/res14", help="Root dir for saving models.")
parser.add_argument("--model", type=str, default="SGAT", help="model to use, (std, GAT, SGAT)")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--bert_out_dim", type=int, default=100)
parser.add_argument("--output_merge",
    type=str,
    default="gatenorm2",
    help="merge method to use, (none, addnorm, add, attn, gate, gatenorm2)",)
parser.add_argument("--max_len", type=int, default=80)

parser.add_argument("--run_name", default=None, type=str, help="run name for mlflow")
parser.add_argument("--logged_model", default=None, type=str, help="model to be used for testing")

args = parser.parse_args()

mlflow.set_tracking_uri("/home/ubuntu/mylrz/atsc-experiments/MODELS/mlruns")
mlflow.set_experiment("rgat-bert")

mlflow.start_run(run_name=args.run_name)

# set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
helper.print_arguments(args)

# load vocab
print("Loading vocab...")
token_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_tok.vocab")  # token
post_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_post.vocab")  # position
pos_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_pos.vocab")  # POS
dep_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_dep.vocab")  # deprel
pol_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_pol.vocab")  # polarity
vocab = (token_vocab, post_vocab, pos_vocab, dep_vocab, pol_vocab)
print(
    "token_vocab: {}, post_vocab: {}, pos_vocab: {}, dep_vocab: {}, pol_vocab: {}".format(
        len(token_vocab), len(post_vocab), len(pos_vocab), len(dep_vocab), len(pol_vocab)
    )
)
args.tok_size = len(token_vocab)
args.post_size = len(post_vocab)
args.pos_size = len(pos_vocab)
args.dep_size = len(dep_vocab)

test_batch = ABSADataLoader(
    args.data_dir + "/test.json", args.batch_size, args, vocab, shuffle=False
)

def evaluate(model, data_loader):
    predictions, labels = [], []
    val_loss, val_acc, val_step = 0.0, 0.0, 0
    id_results = {}
    for i, batch in enumerate(data_loader):
        if "ARTS" in args.data_dir:
            loss, acc, pred, label, _, _ = model.predict_arts(batch)
            ids = batch[-1]

            # for ARS sort sentence according to their ids
            for (id, predicted, lab) in zip(ids, pred, label):

                id = str(id.numpy())

                if id in id_results.keys():
                    id_results[id]["correct"] += int(lab == predicted)
                    id_results[id]["total"] += 1
                else:
                    id_results[id] = {}
                    id_results[id]["correct"] = int(lab == predicted)
                    id_results[id]["total"] = 1
        else:
            loss, acc, pred, label, _, _ = model.predict(batch)

        val_loss += loss
        val_acc += acc
        predictions += pred
        labels += label
        val_step += 1

    if len(id_results) != 0:
        # Aspect Robustness Score (ARS)
        id_preds = []
        id_total = len(id_results.keys()) * [1]

        for id in id_results.keys():
            if id_results[id]["correct"] == id_results[id]["total"]:
                id_preds += [1]
            else:
                id_preds += [0]

        acc_ars = metrics.accuracy_score(id_total, id_preds)

    else:
        acc_ars = -1000

    # f1 scores
    f1_score_macro = metrics.f1_score(labels, predictions, average="macro")
    f1_score_micro = metrics.f1_score(labels, predictions, average="micro")
    f1_score_weighted = metrics.f1_score(labels, predictions, average="weighted")
    # original accuracy does NOT yield the same results as sklearn
    acc_sk = metrics.accuracy_score(labels, predictions)
    return val_loss / val_step, val_acc / val_step, f1_score_macro, f1_score_micro, f1_score_weighted, acc_sk, acc_ars

for arg in vars(args):
    print('>>> {0}: {1}'.format(arg, getattr(args, arg)))
    mlflow.log_param(arg, getattr(args, arg))


mlflow.log_param("logged_model", args.logged_model)

trainer = ABSATrainer(args)
trainer.model = mlflow.pytorch.load_model(args.logged_model)

test_loss, test_acc, test_f1_macro, test_f1_micro, test_f1_weighted, test_acc_sk, test_acc_ars = evaluate(trainer, test_batch)
mlflow.log_metrics({"acc": test_acc_sk, "f1_macro": test_f1_macro, "f1_micro": test_f1_micro, "f1_weighted": test_f1_weighted, "acc_ars": test_acc_ars})
print("Evaluation Results: test_loss:{}, test_acc:{}, test_f1_macro:{}, test_acc_sk: {}, test_acc_ars: {}".format(test_loss, test_acc/100, test_f1_macro, test_acc_sk, test_acc_ars))

mlflow.end_run()