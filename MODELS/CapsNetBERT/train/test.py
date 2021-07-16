import torch
import os
from train import make_aspect_term_model, make_aspect_category_model
from train.make_data import make_term_test_data, make_category_test_data
from train.eval import eval
import mlflow

def test(config, model, best_model_dict=None):

    print("Start Testing ...")

    if best_model_dict == None:

        #if mode == 'term':
        #    model = make_aspect_term_model.make_model(config)
        #else:
        #    model = make_aspect_category_model.make_model(config)
        #model = model.cuda()
        #model_path = os.path.join(config['base_path'], 'checkpoints/%s.pth' % config['aspect_' + mode + '_model']['type'])
        #model.load_state_dict(torch.load(model_path))

        mlflow.log_param("model", model)
        model = mlflow.pytorch.load_model(model)
    else:
        model.load_state_dict(best_model_dict)

    model = model.cuda()

    mode = config['mode']
    if mode == 'term':
        test_loader = make_term_test_data(config)
    else:
        test_loader = make_category_test_data(config)
    test_acc, test_f1_mi, test_f1_ma, test_f1_w = eval(model, test_loader)

    print('test:\taccuracy: %.4f' % (test_acc))
    print('test:\tf1micro: %.4f' % (test_f1_mi))
    print('test:\tf1macro: %.4f' % (test_f1_ma))
    print('test:\tf1weighted: %.4f' % (test_f1_w))

    mlflow.log_metric("accuracy", test_acc)
    mlflow.log_metric("f1_micro", test_f1_mi)
    mlflow.log_metric("f1_macro", test_f1_ma)
    mlflow.log_metric("f1_weighted", test_f1_w)
