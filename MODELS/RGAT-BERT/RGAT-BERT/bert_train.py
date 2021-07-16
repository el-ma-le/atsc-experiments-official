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
import copy

mlflow.set_tracking_uri("/home/ubuntu/mylrz/atsc-experiments/MODELS/mlruns")
mlflow.set_experiment("rgat-bert")

mlflow.start_run()

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
    help="merge method to use, (none, addnorm, add, attn, gate, gatenorm2)",
)
parser.add_argument("--max_len", type=int, default=80)
parser.add_argument("--train_seed", type=int, default=0, help="select train/val split for semeval data")
parser.add_argument("--patience", type=int, default=5, help="maximum number of epochs with no improvement in training")

args = parser.parse_args()

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

# load data
print("Loading data from {} with batch size {}...".format(args.data_dir, args.batch_size))
if args.train_seed == 0:
    train_batch = ABSADataLoader(
        args.data_dir + "/train.json", args.batch_size, args, vocab, shuffle=True
    )
    valid_batch = ABSADataLoader(
        args.data_dir + "/val.json", args.batch_size, args, vocab, shuffle=False
    )
else:
    train_batch = ABSADataLoader(
        args.data_dir + "/train_" + str(args.train_seed) + ".json", args.batch_size, args, vocab, shuffle=True
    )
    valid_batch = ABSADataLoader(
        args.data_dir + "/val_" + str(args.train_seed) + ".json", args.batch_size, args, vocab, shuffle=False
    )
test_batch = ABSADataLoader(
    args.data_dir + "/test.json", args.batch_size, args, vocab, shuffle=False
)


def evaluate(model, data_loader):
    predictions, labels = [], []
    val_loss, val_acc, val_step = 0.0, 0.0, 0
    for i, batch in enumerate(data_loader):
        loss, acc, pred, label, _, _ = model.predict(batch)
        val_loss += loss
        val_acc += acc
        predictions += pred
        labels += label
        val_step += 1
    # f1 scores
    f1_score_macro = metrics.f1_score(labels, predictions, average="macro")
    f1_score_micro = metrics.f1_score(labels, predictions, average="micro")
    f1_score_weighted = metrics.f1_score(labels, predictions, average="weighted")
    # original accuracy does NOT yield the same results as sklearn
    acc_sk = metrics.accuracy_score(labels, predictions)
    return val_loss / val_step, val_acc / val_step, f1_score_macro, f1_score_micro, f1_score_weighted, acc_sk


def _totally_parameters(model):  #
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params

for arg in vars(args):
    print('>>> {0}: {1}'.format(arg, getattr(args, arg)))
    mlflow.log_param(arg, getattr(args, arg))


# build model
trainer = ABSATrainer(args)
print('# parameters:', _totally_parameters(trainer.model))
print("Training Set: {}".format(len(train_batch)))
print("Valid Set: {}".format(len(valid_batch)))
print("Test Set: {}".format(len(test_batch)))

val_acc_sk_history = [0.0]
val_f1_macro_history = [0.0]
max_val_epoch = 0

for epoch in range(1, args.num_epoch + 1):
    print("Epoch {}".format(epoch) + "-" * 60)
    train_loss, train_acc, train_step = 0.0, 0.0, 0
    for i, batch in enumerate(train_batch):
        loss, acc = trainer.update(batch)
        train_loss += loss
        train_acc += acc
        train_step += 1
        if train_step % args.log_step == 0:
            print("{}/{} train_loss: {:.6f}, train_acc: {:.6f}".format(
                    i, len(train_batch), train_loss / train_step, train_acc / (100*train_step)))
            mlflow.log_metrics({"train_loss": np.float64(train_loss/train_step), "train_acc": np.float64(train_acc/(100*train_step))}, i)

    val_loss, val_acc, val_f1_macro, _, _, val_acc_sk = evaluate(trainer, valid_batch)
    mlflow.log_metrics({"val_loss": np.float64(val_loss), "val_acc_sk": np.float64(val_acc_sk), "val_f1_macro": np.float64(val_f1_macro)}, epoch)

    print("End of epoch {}: train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc_sk: {:.4f}, val_f1_macro: {:.4f}".format(
            epoch, train_loss / train_step, train_acc / (100*train_step), val_loss, val_acc_sk, val_f1_macro))


    if epoch == 1 or float(val_acc_sk) >= max(val_acc_sk_history):
        best_model = copy.deepcopy(trainer.model)
        best_model_dict = copy.deepcopy(trainer.model.state_dict())
        max_val_epoch = epoch

    if epoch == 1 or float(val_f1_macro) >= max(val_f1_macro_history):
        best_model = copy.deepcopy(trainer.model)
        best_model_dict = copy.deepcopy(trainer.model.state_dict())
        max_val_epoch = epoch

    val_acc_sk_history.append(float(val_acc_sk))
    val_f1_macro_history.append(float(val_f1_macro))

    if epoch - max_val_epoch >= args.patience:
        print('>> early stop.')
        mlflow.log_metric("early_stopp_at_epoch", epoch)
        break

mlflow.pytorch.log_model(best_model, "model")

trainer.load_state_dict_only(best_model_dict)

test_loss, test_acc, test_f1_macro, test_f1_micro, test_f1_weighted, test_acc_sk = evaluate(trainer, test_batch)
mlflow.log_metrics({"acc": test_acc_sk, "f1_macro": test_f1_macro, "f1_micro": test_f1_micro, "f1_weighted": test_f1_weighted})
print("Evaluation Results: test_loss:{}, test_acc:{}, test_f1_macro:{}, test_acc_sk: {}".format(test_loss, test_acc/100, test_f1_macro, test_acc_sk))

mlflow.end_run()