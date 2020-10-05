import time
from copy import deepcopy

import numpy as np
import torch
from pytorch_pretrained_bert.optimization import BertAdam
from sklearn.metrics import accuracy_score, classification_report
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from dataset import class_list


class Stat:
    def __init__(self, training, writer=None):
        self.step = 0
        self.loss = []
        self.labels = []
        self.pred_labels = []
        self.training = training
        self.writer = writer

    def add(self, pred, gold_labels, loss):
        gold_labels = gold_labels.cpu().numpy()
        pred = pred.cpu().detach().numpy()
        pred_labels = np.argmax(pred, axis=1)

        self.loss.append(loss)
        self.labels.extend(gold_labels)
        self.pred_labels.extend(pred_labels)

    def log(self):
        self.step += 1
        acc = accuracy_score(self.labels, self.pred_labels)
        loss = sum(self.loss) / len(self.loss)
        self.loss = []
        self.labels = []
        self.pred_labels = []
        if not self.writer:
            return loss, acc
        if self.training:
            self.writer.add_scalar('train_Loss', loss, self.step)
            self.writer.add_scalar('train_Accuracy', acc, self.step)
        else:
            self.writer.add_scalar('dev_Loss', loss, self.step)
            self.writer.add_scalar('dev_Accuracy', acc, self.step)
        return loss, acc


def train(args, model, train_data_loader, dev_data_loader):
    loss_func = CrossEntropyLoss()

    if args['type'] == "BERT":
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args['lr'],
                             warmup=0.05,
                             t_total=len(train_data_loader) *args['num_epochs'],
                             weight_decay=args['weight_decay'])
    else:
        optimizer = Adam(model.parameters(), args['lr'], weight_decay=args['weight_decay'])

    writer = SummaryWriter(args['output_path']+'/' +
                           time.strftime('%m-%d_%H.%M', time.localtime()))
    train_stat = Stat(training=True, writer=writer)
    dev_stat = Stat(training=False, writer=writer)

    best_acc, best_net = 0, None

    for epoch in range(args['num_epochs']):
        print(f"--- epoch: {epoch + 1} ---")

        for iter, batch in enumerate(train_data_loader):
            model.train()
            inputs, labels = batch[0], batch[1]
            optimizer.zero_grad()
            pred_outputs = model(inputs)

            loss = loss_func(pred_outputs, labels)
            loss.backward()
            optimizer.step()
            train_stat.add(pred_outputs, labels, loss.item())

            if (iter + 1) % args['display_per_batch'] == 0:
                train_loss, train_acc = train_stat.log()

                model.eval()
                with torch.no_grad():
                    for batch in dev_data_loader:
                        inputs, labels = batch[0], batch[1]
                        pred_outputs = model(inputs)

                        loss = loss_func(pred_outputs, labels)

                        dev_stat.add(pred_outputs, labels, loss.item())
                dev_loss, dev_acc = dev_stat.log()
                print(f"step {(iter + 1):5}, "\
                    f"training loss: {train_loss:.4f}, acc: {train_acc:.2%}, "\
                    f"dev loss: {dev_loss:.4f}, acc: {dev_acc:.2%}.")
                
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    best_net = deepcopy(model.state_dict())
                

    print(f"best dev acc: {best_acc:.4f}")
    return best_net


def test(model, test_data_loader):
    test_stat = Stat(training=False)
    print("--- testing ---")
    model.eval()
    with torch.no_grad():
        for batch in test_data_loader:
            inputs, labels = batch[0], batch[1]
            pred_outputs = model(inputs)
            test_stat.add(pred_outputs, labels, 0)  # dummy for loss

    report = classification_report(
        test_stat.labels, 
        test_stat.pred_labels, 
        target_names=class_list, 
        digits=4, zero_division=0)
    print(report)
