import torch
import sklearn.metrics

def eval(model, data_loader, criterion=None):
    total_samples = 0
    correct_samples = 0
    total_loss = 0
    all_labels = []
    all_preds = []
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            input0, input1, label = data
            input0, input1, label = input0.cuda(), input1.cuda(), label.cuda()
            all_labels.extend(label.detach().cpu().numpy())
            logit = model(input0, input1)
            loss = criterion(logit, label).item() if criterion is not None else 0
            total_samples += input0.size(0)
            pred = logit.argmax(dim=1)
            all_preds.extend(pred.detach().cpu().numpy())
            correct_samples += (label == pred).long().sum().item()
            total_loss += loss * input0.size(0)
    #accuracy = correct_samples / total_samples # returns the same value as sklearn function
    avg_loss = total_loss / total_samples

    acc_sk = sklearn.metrics.accuracy_score(all_labels, all_preds)
    f1_micro = sklearn.metrics.f1_score(all_labels, all_preds, average="micro")
    f1_macro = sklearn.metrics.f1_score(all_labels, all_preds, average="macro")
    f1_weighted = sklearn.metrics.f1_score(all_labels, all_preds, average="weighted")

    if criterion is not None:
        return acc_sk, avg_loss, f1_macro
    else:
        return acc_sk, f1_micro, f1_macro, f1_weighted
