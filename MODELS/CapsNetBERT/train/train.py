import torch
from torch import nn
from torch import optim
from train import make_aspect_term_model, make_aspect_category_model
from train.make_data import make_term_data, make_category_data
from train.make_optimizer import make_optimizer
from train.eval import eval
import os
import time
import pickle
from src.module.utils.loss import CapsuleLoss
import mlflow
import copy
import numpy as np
import random 

def train(config):

    if config['seed'] is not None:
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])

    mode = config['mode']
    if mode == 'term':
        model = make_aspect_term_model.make_model(config)
        train_loader, val_loader = make_term_data(config)
    else:
        model = make_aspect_category_model.make_model(config)
        train_loader, val_loader = make_category_data(config)

    model = model.cuda()
    base_path = config['base_path']
    model_path = os.path.join(base_path, 'checkpoints/%s.pth' % config['aspect_' + mode + '_model']['type'])
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    with open(os.path.join(base_path, 'processed/index2word.pickle'), 'rb') as handle:
        index2word = pickle.load(handle)

    criterion = CapsuleLoss()
    optimizer = make_optimizer(config, model)
    max_val_accuracy = 0
    max_val_f1_macro = 0
    max_val_epoch = 0
    #min_val_loss = 100
    global_step = 0
    patience = config['patience']

    config = config['aspect_' + mode + '_model'][config['aspect_' + mode + '_model']['type']]
    print("Start Training ...")
    for epoch in range(config['num_epoches']):
        print("Current Epoch: ", epoch)
        total_loss = 0
        total_samples = 0
        correct_samples = 0
        start = time.time()
        for i, data in enumerate(train_loader):
            global_step += 1
            model.train()
            input0, input1, label = data
            input0, input1, label = input0.cuda(), input1.cuda(), label.cuda()
            optimizer.zero_grad()
            logit = model(input0, input1)
            loss = criterion(logit, label)
            batch_size = input0.size(0)
            total_loss += batch_size * loss.item()
            total_samples += batch_size
            pred = logit.argmax(dim=1)
            correct_samples += (label == pred).long().sum().item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if i % 10 == 0 and i > 0:
                train_loss = total_loss / total_samples
                train_accuracy = correct_samples / total_samples
                mlflow.log_metric("train_loss", train_loss, i)
                mlflow.log_metric("train_acc", train_accuracy, i)
                total_loss = 0
                total_samples = 0
                correct_samples = 0

        val_accuracy, val_loss, val_f1_macro = eval(model, val_loader, criterion)
        mlflow.log_metric("val_loss", val_loss, epoch)
        mlflow.log_metric("val_acc", val_accuracy, epoch)
        mlflow.log_metric("val_f1_macro", val_f1_macro, epoch)
        print('[epoch %2d] val_loss: %.4f val_acc: %.4f'
                % (epoch, val_loss, val_accuracy))

        if val_accuracy >= max_val_accuracy:
            max_val_accuracy = val_accuracy
            max_val_epoch = epoch
            best_model = copy.deepcopy(model)
            best_model_dict = copy.deepcopy(model.state_dict())
        if val_f1_macro >= max_val_f1_macro:
            max_val_f1_macro = val_f1_macro
            max_val_epoch = epoch
            best_model = copy.deepcopy(model)
            best_model_dict = copy.deepcopy(model.state_dict())
        if epoch - max_val_epoch >= patience:
            print('>> early stop.')
            mlflow.log_metric("early_stopp_at_epoch", epoch)
            break

        end = time.time()
        print('time: %.4fs' % (end - start))

    print('max_val_accuracy:', max_val_accuracy)
    print('max_val_f1_macro:', max_val_f1_macro)
    mlflow.pytorch.log_model(best_model, "model")

    return model, best_model_dict