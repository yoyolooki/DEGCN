# -*- coding: utf-8 -*-
# @Time    : 2024/5/4 19:07
# @Author  : Li Yu
# @File    : dense_gcn_model.py
import numpy as np
import pandas as pd
import argparse
import glob
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import torch.nn.functional as F
from dense_gcn_model import DEGCN
from utils import load_data
from utils import accuracy
import matplotlib.pyplot as plt


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train(epoch, optimizer, features, adj, labels, idx_train):
    '''
    :param epoch: training epochs
    :param optimizer: training optimizer, Adam optimizer
    :param features: the omics features
    :param adj: the laplace adjacency matrix
    :param labels: sample labels
    :param idx_train: the index of trained samples
    '''
    labels.to(device)

    DEGCN_model.train()
    optimizer.zero_grad()
    output = DEGCN_model(features, adj)

    # Calculate cross-entropy loss on the training set
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print('Epoch: %.2f | loss train: %.4f | acc train: %.4f' % (epoch + 1, loss_train.item(), acc_train.item()))
    return loss_train.data.item()


def test(features, adj, labels, idx_test):
    '''
    :param features: the omics features
    :param adj: the laplace adjacency matrix
    :param labels: sample labels
    :param idx_test: the index of tested samples
    '''
    DEGCN_model.eval()
    with torch.no_grad():
        output = DEGCN_model(features, adj)
        loss_test = F.cross_entropy(output[idx_test], labels[idx_test])

        # Calculate the accuracy
        _, predictions = torch.max(output[idx_test], dim=1)
        acc_test = (predictions == labels[idx_test]).float().mean()

        # Convert tensors to numpy arrays for metric calculations
        predictions = predictions.cpu().numpy()
        true_labels = labels[idx_test].cpu().numpy()

        # Calculate Precision, Recall, and F1 Score
        precision = precision_score(true_labels, predictions, average='weighted')
        recall = recall_score(true_labels, predictions, average='weighted')
        f1 = f1_score(true_labels, predictions, average='weighted')

        print('Predicted labels:', predictions)
        print('Original labels:', true_labels)

        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()),
              "precision= {:.4f}".format(precision),
              "recall= {:.4f}".format(recall),
              "f1_score= {:.4f}".format(f1))

        # Return accuracy, f1 score, precision, and recall
        return acc_test.item(), f1, precision, recall


def predict(features, adj, sample, idx):
    '''
    :param features: the omics features
    :param adj: the laplace adjacency matrix
    :param sample: all sample names
    :param idx: the index of predict samples
    :return:
    '''
    DEGCN_model.eval()
    output = DEGCN_model(features, adj)
    predict_label = output.detach().cpu().numpy()
    predict_label = np.argmax(predict_label, axis=1).tolist()
    # print(predict_label)

    res_data = pd.DataFrame({'Sample': sample, 'predict_label': predict_label})
    res_data = res_data.iloc[idx, :]
    # print(res_data)

    res_data.to_csv('result/gcn_predicted_data.csv', header=True, index=False)


def plot_loss_curve(all_loss_values, smoothing_factor=0.9):
    plt.figure()
    for i, loss_values in enumerate(all_loss_values):
        smoothed_loss = [loss_values[0]]
        for j in range(1, len(loss_values)):
            smoothed_value = smoothing_factor * smoothed_loss[-1] + (1 - smoothing_factor) * loss_values[j]
            smoothed_loss.append(smoothed_value)
        plt.plot(smoothed_loss, label=f'Fold {i + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig('result/gcn_train_loss.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--featuredata', '-fd', type=str, required=True, help='The vector feature file.')
    parser.add_argument('--adjdata', '-ad', type=str, required=True, help='The adjacency matrix file.')
    parser.add_argument('--labeldata', '-ld', type=str, required=True, help='The sample label file.')
    parser.add_argument('--testsample', '-ts', type=str, help='Test sample names file.')
    parser.add_argument('--mode', '-m', type=int, choices=[0, 1], default=0,
                        help='mode 0: 10-fold cross validation; mode 1: train and test a model.')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed, default=0.')
    parser.add_argument('--device', '-d', type=str, choices=['cpu', 'gpu'], default='cpu',
                        help='Training on cpu or gpu, default: cpu.')
    parser.add_argument('--epochs', '-e', type=int, default=150, help='Training epochs, default: 150.')
    parser.add_argument('--learningrate', '-lr', type=float, default=0.001, help='Learning rate, default: 0.001.')
    parser.add_argument('--weight_decay', '-w', type=float, default=0.01,
                        help='Weight decay (L2 loss on parameters), methods to avoid overfitting, default: 0.01')
    parser.add_argument('--hidden', '-hd', type=int, default=64, help='Hidden layer dimension, default: 64.')
    parser.add_argument('--dropout', '-dp', type=float, default=0.5,
                        help='Dropout rate, methods to avoid overfitting, default: 0.5.')
    parser.add_argument('--threshold', '-t', type=float, default=0.005,
                        help='Threshold to filter edges, default: 0.005')
    parser.add_argument('--nclass', '-nc', type=int, default=4, help='Number of classes, default: 4')
    parser.add_argument('--patience', '-p', type=int, default=20, help='Patience')
    args = parser.parse_args()

    # Check whether GPUs are available
    device = torch.device('cpu')
    if args.device == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set random seed
    setup_seed(args.seed)

    # load input files
    adj, data, label = load_data(args.adjdata, args.featuredata, args.labeldata, args.threshold)

    # change dataframe to Tensor
    adj = torch.tensor(adj, dtype=torch.float, device=device)
    features = torch.tensor(data.iloc[:, 1:].values, dtype=torch.float, device=device)
    labels = torch.tensor(label.iloc[:, 1].values, dtype=torch.long, device=device)

    print('Begin training model...')

    # 10-fold cross validation
    if args.mode == 0:
        skf = StratifiedKFold(n_splits=10, shuffle=True)

        acc_res, f1_res, prec_res, rec_res = [], [], [], []  # record accuracy, f1 score, precision, and recall
        all_loss_values = []  # record all loss values for each fold

        # split train and test data
        for (idx_train, idx_test) in skf.split(data.iloc[:, 1:], label.iloc[:, 1]):
            # initialize a model
            DEGCN_model = DEGCN(n_in=features.shape[1], n_hid=args.hidden, n_out=args.nclass, dropout=args.dropout)
            DEGCN_model.to(device)

            # optimizer
            optimizer = torch.optim.Adam(DEGCN_model.parameters(), lr=args.learningrate, weight_decay=args.weight_decay)

            idx_train, idx_test = torch.tensor(idx_train, dtype=torch.long, device=device), torch.tensor(idx_test,
                                                                                                         dtype=torch.long,
                                                                                                         device=device)

            fold_loss_values = []
            for epoch in range(args.epochs):
                loss = train(epoch, optimizer, features, adj, labels, idx_train)
                fold_loss_values.append(loss)
            all_loss_values.append(fold_loss_values)

            # Calculate the accuracy, f1 score, precision, and recall
            ac, f1, prec, rec = test(features, adj, labels, idx_test)

            acc_res.append(ac)
            f1_res.append(f1)
            prec_res.append(prec)
            rec_res.append(rec)

        # Print results
        print('10-fold  Acc(%.4f, %.4f)  F1(%.4f, %.4f)  Precision(%.4f, %.4f)  Recall(%.4f, %.4f)' % (
            np.mean(acc_res), np.std(acc_res),
            np.mean(f1_res), np.std(f1_res),
            np.mean(prec_res), np.std(prec_res),
            np.mean(rec_res), np.std(rec_res)))

        predict(features, adj, data['Sample'].tolist(), data.index.tolist())

        # Plot the training loss curve for the last fold
        plot_loss_curve(all_loss_values)

    elif args.mode == 1:
        # load test samples
        test_sample_df = pd.read_csv(args.testsample, header=0, index_col=None)
        test_sample = test_sample_df.iloc[:, 0].tolist()
        all_sample = data['Sample'].tolist()
        train_sample = list(set(all_sample) - set(test_sample))

        # get index of train samples and test samples
        train_idx = data[data['Sample'].isin(train_sample)].index.tolist()
        test_idx = data[data['Sample'].isin(test_sample)].index.tolist()
        print('features.shape:', features.shape[1])

        # create DEGCN
        DEGCN_model = DEGCN(n_in=features.shape[1], n_hid=args.hidden, n_out=args.nclass, dropout=args.dropout)
        DEGCN_model.to(device)
        optimizer = torch.optim.Adam(DEGCN_model.parameters(), lr=args.learningrate, weight_decay=args.weight_decay)
        idx_train, idx_test = torch.tensor(train_idx, dtype=torch.long, device=device), torch.tensor(test_idx,
                                                                                                     dtype=torch.long,
                                                                                                     device=device)

        '''
        save a best model (with the minimum loss value)
        if the loss didn't decrease in N epochs，stop the train process.
        N can be set by args.patience 
        '''
        loss_values = []  # record the loss value of each epoch
        # record the times with no loss decrease, record the best epoch
        bad_counter, best_epoch = 0, 0
        best = 1000  # record the lowest loss value
        for epoch in range(args.epochs):
            loss_values.append(train(epoch, optimizer, features, adj, labels, idx_train))
            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1  # In this epoch, the loss value didn't decrease

            if bad_counter == args.patience:
                break

            # save model of this epoch
            torch.save(DEGCN_model.state_dict(), 'model/DEGCN/{}.pkl'.format(epoch))

            # reserve the best model, delete other models
            files = glob.glob('model/DEGCN/*.pkl')
            for file in files:
                name = file.split('\\')[1]
                epoch_nb = int(name.split('.')[0])
                # print(file, name, epoch_nb)
                if epoch_nb != best_epoch:
                    os.remove(file)

        print('Training finished.')
        print('The best epoch model is ', best_epoch)
        DEGCN_model.load_state_dict(torch.load('model/DEGCN/{}.pkl'.format(best_epoch)))
        # predict(features, adj, all_sample, test_idx)

    print('Finished!')
