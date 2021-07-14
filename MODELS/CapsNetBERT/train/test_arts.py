import torch
import os
from train.make_data import make_term_test_data_arts
from train.eval_arts import eval
import mlflow

def test(config, model, best_model_dict=None):

    print("Start Testing ...")

    if best_model_dict == None:
        mlflow.log_param("model", model)
        model = mlflow.pytorch.load_model(model)
    else:
        model.load_state_dict(best_model_dict)

    model = model.cuda()

    mode = config['mode']
    if mode == 'term':
        test_loader = make_term_test_data_arts(config)
    test_acc, test_f1_mi, test_f1_ma, test_f1_w, test_acc_ars = eval(model, test_loader)

    print('test:\taccuracy: %.4f' % (test_acc))
    print('test:\tf1micro: %.4f' % (test_f1_mi))
    print('test:\tf1macro: %.4f' % (test_f1_ma))
    print('test:\tf1weighted: %.4f' % (test_f1_w))
    print('test:\taccuracy_ars: %.4f' % (test_acc_ars))

    mlflow.log_metric("accuracy", test_acc)
    mlflow.log_metric("f1_micro", test_f1_mi)
    mlflow.log_metric("f1_macro", test_f1_ma)
    mlflow.log_metric("f1_weighted", test_f1_w)
    mlflow.log_metric("accuracy_ars", test_acc_ars)
