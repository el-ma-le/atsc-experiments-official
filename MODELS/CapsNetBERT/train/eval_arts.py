import torch
import sklearn.metrics
import numpy as np

def eval(model, data_loader, criterion=None):
    total_samples = 0
    correct_samples = 0
    total_loss = 0
    all_labels = []
    all_preds = []
    id_results = {}
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            input0, input1, ids, label = data
            input0, input1, label = input0.cuda(), input1.cuda(), label.cuda()
            all_labels.extend(label.detach().cpu().numpy())
            logit = model(input0, input1)
            loss = criterion(logit, label).item() if criterion is not None else 0
            total_samples += input0.size(0)
            pred = logit.argmax(dim=1)
            all_preds.extend(pred.detach().cpu().numpy())
            correct_samples += (label == pred).long().sum().item()
            total_loss += loss * input0.size(0)

            # for ARS sort sentence according to their ids
            for (id, predicted, lab) in zip(ids, pred.detach().cpu().numpy(), label.detach().cpu().numpy()):

                id = str(id.detach().cpu().numpy())

                if id in id_results.keys():
                    id_results[id]["correct"] += np.int(lab == predicted)
                    id_results[id]["total"] += 1
                else:
                    id_results[id] = {}
                    id_results[id]["correct"] = np.int(lab == predicted)
                    id_results[id]["total"] = 1

    # Aspect Robustness Score (ARS)
    id_preds = []
    id_total = len(id_results.keys()) * [1]

    for id in id_results.keys():
        if id_results[id]["correct"] == id_results[id]["total"]:
            id_preds += [1]
        else:
            id_preds += [0]
    
    acc_ars = sklearn.metrics.accuracy_score(id_total, id_preds)

    #accuracy = correct_samples / total_samples # returns the same value as sklearn function
    avg_loss = total_loss / total_samples

    acc_sk = sklearn.metrics.accuracy_score(all_labels, all_preds)
    f1_micro = sklearn.metrics.f1_score(all_labels, all_preds, average="micro")
    f1_macro = sklearn.metrics.f1_score(all_labels, all_preds, average="macro")
    f1_weighted = sklearn.metrics.f1_score(all_labels, all_preds, average="weighted")

    if criterion is not None:
        return acc_sk, avg_loss, f1_macro
    else:
        return acc_sk, f1_micro, f1_macro, f1_weighted, acc_ars
