from __future__ import division
from __future__ import print_function
from sklearn import metrics
import time
import sys

import torch
import torch.nn as nn

import numpy as np

from utils import *
from gcn import GCN

from config import CONFIG

cfg = CONFIG()

if len(sys.argv) != 2:
    sys.exit("Use: python train.py <dataset>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'r8']
dataset = sys.argv[1]
class_list = [x.strip()
              for x in open('..\data\class.txt', encoding='utf8').readlines()]

if dataset not in datasets:
    sys.exit("wrong dataset name")
cfg.dataset = dataset

# Set random seed
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)


# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(cfg.dataset)

features = sp.identity(features.shape[0])  # featureless


# Some preprocessing
features = preprocess_features(features)
if cfg.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif cfg.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, cfg.max_degree)
    num_supports = 1 + cfg.max_degree
    model_func = GCN
else:
    raise ValueError('Invalid argument for model: ' + str(cfg.model))

# Define placeholders
t_features = torch.from_numpy(features)
t_y_train = torch.from_numpy(y_train)
t_y_val = torch.from_numpy(y_val)
t_y_test = torch.from_numpy(y_test)
t_train_mask = torch.from_numpy(train_mask.astype(np.float32))
tm_train_mask = torch.transpose(torch.unsqueeze(
    t_train_mask, 0), 1, 0).repeat(1, y_train.shape[1])

t_support = []
for i in range(len(support)):
    t_support.append(torch.Tensor(support[i]))

model = model_func(input_dim=features.shape[0], support=t_support, num_classes=y_train.shape[1])

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)


# Define model evaluation function
def evaluate(features, labels, mask):
    t_test = time.time()
    model.eval()
    with torch.no_grad():
        logits = model(features)
        t_mask = torch.from_numpy(np.array(mask * 1., dtype=np.float32))
        tm_mask = torch.transpose(torch.unsqueeze(
            t_mask, 0), 1, 0).repeat(1, labels.shape[1])
        loss = criterion(logits * tm_mask, torch.max(labels, 1)[1])
        pred = torch.max(logits, 1)[1]
        acc = ((pred == torch.max(labels, 1)[1]).float(
        ) * t_mask).sum().item() / t_mask.sum().item()

    return loss.numpy(), acc, pred.numpy(), labels.numpy(), (time.time() - t_test)


val_losses = []

# Train model
for epoch in range(cfg.epochs):

    t = time.time()

    # Forward pass
    logits = model(t_features)
    loss = criterion(logits * tm_train_mask, torch.max(t_y_train, 1)[1])
    acc = ((torch.max(logits, 1)[1] == torch.max(t_y_train, 1)[
        1]).float() * t_train_mask).sum().item() / t_train_mask.sum().item()

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    val_loss, val_acc, pred, labels, duration = evaluate(
        t_features, t_y_val, val_mask)
    val_losses.append(val_loss)

    print(f"Epoch: {epoch+1:.0f}, train_loss: {loss:.4f}, train_acc: {acc:.2%}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.2%}")

    if epoch > cfg.early_stopping and val_losses[-1] > np.mean(val_losses[-(cfg.early_stopping + 1):-1]):
        print("Early stopping...")
        break

test_loss, test_acc, pred, labels, test_duration = evaluate(t_features, t_y_test, test_mask)
# Testing
test_pred = []
test_labels = []
for i in range(len(test_mask)):
    if test_mask[i]:
        test_pred.append(pred[i])
        test_labels.append(np.argmax(labels[i]))

print(metrics.classification_report(
    test_labels, test_pred, digits=4, zero_division=0, target_names=class_list))
