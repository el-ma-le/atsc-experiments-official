# -*- coding: utf-8 -*-
# file: train.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import argparse
import math
import os
import sys
import random
import numpy

from sklearn import metrics
from time import strftime, localtime

from transformers import BertModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset
from models import LSTM, IAN, MemNet, RAM, TD_LSTM, TC_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, ASGCN, LCF_BERT
from models.aen import CrossEntropyLoss_LSR, AEN_BERT
from models.bert_spc import BERT_SPC

import mlflow
import copy

mlflow.set_tracking_uri("/home/ubuntu/mylrz/atsc-experiments/MODELS/mlruns")
mlflow.set_experiment("mgatn")


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        if 'bert' in opt.model_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['val']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_split{1}_tokenizer.dat'.format(opt.dataset, opt.train_seed))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_split{2}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset, opt.train_seed))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer)
        self.valset = ABSADataset(opt.dataset_file['val'], tokenizer)
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer)

        if opt.device.type == 'cuda':
            print('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))
            mlflow.log_param(arg, getattr(self.opt, arg))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0
        global_step = 0
        path = None
        for i_epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = batch['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                train_acc = n_correct / n_total
                train_loss = loss_total / n_total
                mlflow.log_metric("train_acc", train_acc, i_batch)
                mlflow.log_metric("train_loss", train_loss, i_batch)

            val_acc, val_f1_macro, _, _  = self._evaluate_acc_f1(val_data_loader)
            print('> val_acc: {:.4f}, val_f1_macro: {:.4f}'.format(val_acc, val_f1_macro))
            mlflow.log_metric("val_acc", val_acc, i_epoch)
            mlflow.log_metric("val_f1_macro", val_f1_macro, i_epoch)

            if val_acc >= max_val_acc:
                max_val_acc = val_acc
                max_val_epoch = i_epoch
                best_model = copy.deepcopy(self.model)
                best_model_dict = copy.deepcopy(self.model.state_dict())
            if val_f1_macro >=  max_val_f1:
                max_val_f1 = val_f1_macro
                max_val_epoch = i_epoch
                best_model = copy.deepcopy(self.model)
                best_model_dict = copy.deepcopy(self.model.state_dict())
            if i_epoch - max_val_epoch >= self.opt.patience:
                print('>> early stop.')
                mlflow.log_metric("early_stopp_at_epoch", i_epoch)
                break

        mlflow.pytorch.log_model(best_model, "model")
        return best_model_dict

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_batch['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1_macro = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        f1_micro = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='micro')
        f1_weighted = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='weighted')
        return acc, f1_macro, f1_micro, f1_weighted

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        best_model_dict = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        #self.model.load_state_dict(torch.load(best_model_path))
        self.model.load_state_dict(best_model_dict)
        test_acc, test_f1_macro, test_f1_micro, test_f1_weighted = self._evaluate_acc_f1(test_data_loader)
        mlflow.log_metrics({"acc": test_acc, "f1_macro": test_f1_macro, "f1_micro": test_f1_micro, "f1_weighted": test_f1_weighted})
        print('>> test_acc: {:.4f}, test_f1_macro: {:.4f}'.format(test_acc, test_f1_macro))


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='mgan', type=str)
    parser.add_argument('--dataset', default='laptop', type=str, help='mams, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=1e-3, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=50, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=85, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=1, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0.1, type=float, help='set ratio between 0 and 1 for validation support')
    # The following two parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int, help='semantic-relative-distance, see the paper of LCF-BERT model')

    parser.add_argument('--train_seed', default=0, type=int, help='select train/val split for semeval data')
    parser.add_argument('--run_name', default=None, type=str, help='run name for mlflow')

    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = {
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'tc_lstm': TC_LSTM,
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'tnet_lf': TNet_LF,
        'aoa': AOA,
        'mgan': MGAN,
        'asgcn': ASGCN,
        'bert_spc': BERT_SPC,
        'aen_bert': AEN_BERT,
        'lcf_bert': LCF_BERT,
        # default hyper-parameters for LCF-BERT model is as follws:
        # lr: 2e-5
        # l2: 1e-5
        # batch size: 16
        # num epochs: 5
    }
    dataset_files = {
        'restaurant': {
            'train': '/home/ubuntu/mylrz/atsc-experiments/DATA/SEMEVAL-14-REST/train_'+str(opt.train_seed)+'.xml.seg',
            'val': '/home/ubuntu/mylrz/atsc-experiments/DATA/SEMEVAL-14-REST/val_'+str(opt.train_seed)+'.xml.seg',
            'test': '/home/ubuntu/mylrz/atsc-experiments/DATA/SEMEVAL-14-REST/test.xml.seg'
        },
        'laptop': {
            'train': '/home/ubuntu/mylrz/atsc-experiments/DATA/SEMEVAL-14-LAP/train_'+str(opt.train_seed)+'.xml.seg',
            'val': '/home/ubuntu/mylrz/atsc-experiments/DATA/SEMEVAL-14-LAP/val_'+str(opt.train_seed)+'.xml.seg',
            'test': '/home/ubuntu/mylrz/atsc-experiments/DATA/SEMEVAL-14-LAP/test.xml.seg'
        },
        'mams': {
            'train': '/home/ubuntu/mylrz/atsc-experiments/DATA/MAMS/train.xml.seg',
            'val': '/home/ubuntu/mylrz/atsc-experiments/DATA/MAMS/val.xml.seg',
            'test': '/home/ubuntu/mylrz/atsc-experiments/DATA/MAMS/test.xml.seg'
        }
    }

    input_colses = {
        'lstm': ['text_indices'],
        'td_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices'],
        'tc_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices', 'aspect_indices'],
        'atae_lstm': ['text_indices', 'aspect_indices'],
        'ian': ['text_indices', 'aspect_indices'],
        'memnet': ['context_indices', 'aspect_indices'],
        'ram': ['text_indices', 'aspect_indices', 'left_indices'],
        'cabasc': ['text_indices', 'aspect_indices', 'left_with_aspect_indices', 'right_with_aspect_indices'],
        'tnet_lf': ['text_indices', 'aspect_indices', 'aspect_boundary'],
        'aoa': ['text_indices', 'aspect_indices'],
        'mgan': ['text_indices', 'aspect_indices', 'left_indices'],
        'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'bert_spc': ['concat_bert_indices', 'concat_segments_indices'],
        'aen_bert': ['text_bert_indices', 'aspect_bert_indices'],
        'lcf_bert': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    with mlflow.start_run(run_name=opt.run_name):
        ins = Instructor(opt)
        ins.run()

if __name__ == '__main__':
    main()
    