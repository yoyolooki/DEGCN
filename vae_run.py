import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
from torch.utils.data import DataLoader
from dataloader import Omics_Data_VAE
from vae_model import VAE
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_vae(model, epochs, criterion, optimizer, train_loader):
    criterion = criterion
    optimizer = optimizer
    train_loader = train_loader

    loss_list = []
    for epoch in range(epochs):
        train_loss = 0.0
        for batch_idx, (omics1_data, omics2_data, omics3_data) in enumerate(train_loader):
            # Get a batch of training data and move it to the device
            omics1_data_input = omics1_data.to(device)
            omics2_data_input = omics2_data.to(device)
            omics3_data_input = omics3_data.to(device)

            # Forward
            latent_data, latent_data_z, decoded1, decoded2, decoded3, mu1, mu2, mu3, log_var1, log_var2, log_var3 = model(
                omics1_data_input, omics2_data_input, omics3_data_input)

            # Compute the loss and perform backpropagation
            KLD_moics1 = -0.5 * torch.sum(1 + log_var1 - mu1.pow(2) - log_var1.exp())
            KLD_moics2 = -0.5 * torch.sum(1 + log_var2 - mu2.pow(2) - log_var2.exp())
            KLD_moics3 = -0.5 * torch.sum(1 + log_var3 - mu3.pow(2) - log_var3.exp())
            loss_omics1 = criterion(decoded1, omics1_data_input) + 3 * KLD_moics1
            loss_omics2 = criterion(decoded2, omics2_data_input) + 3 * KLD_moics2
            loss_omics3 = criterion(decoded3, omics3_data_input) + 3 * KLD_moics3
            loss = 0.4 * loss_omics1 + 0.3 * loss_omics2 + 0.3 * loss_omics3

            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪：防止梯度爆炸，设置梯度裁剪值
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update the running loss
            train_loss += loss.item()

        loss_list.append(train_loss)

        # Print the epoch loss
        epoch_train_loss = train_loss / len(train_loader.dataset)

        print(
            "Epoch {}/{}: epoch_train_loss={:.4f}".format(epoch + 1, epochs, epoch_train_loss)
        )

    # draw the training loss curve
    plt.plot([i + 1 for i in range(epochs)], loss_list)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('VAE Training Loss')
    plt.savefig('result/vae_train_loss.png')
    plt.close()


def work(epochs, num_hidden, mode, path1, path2, path3, device, lr, batch_size):
    # train VAE model
    if mode == 0 or mode == 1:
        print('Training model...')
        # data
        omics_dataset = Omics_Data_VAE(path1, path2, path3)
        train_loader = DataLoader(dataset=omics_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0)  # num_workers多线程

        omics1_dim = omics_dataset[0][0].size(0)
        omics2_dim = omics_dataset[0][1].size(0)
        omics3_dim = omics_dataset[0][2].size(0)

        # model
        vae_model = VAE(num_hidden=num_hidden, omics1_dim=omics1_dim, omics2_dim=omics2_dim, omics3_dim=omics3_dim)
        vae_model.to(device)

        # loss and optimizer
        criterion = nn.MSELoss(reduction="sum")
        optimizer = torch.optim.Adam(vae_model.parameters(), lr=lr)

        # train
        train_vae(epochs=epochs, model=vae_model, criterion=criterion,
                  optimizer=optimizer, train_loader=train_loader)

        vae_model.eval()
        model_path = "model/VAE/VAE_model_for_gcdata.pkl"
        torch.save(vae_model, model_path)

    # load saved model, used for reducing dimensions
    if mode == 0 or mode == 2:
        print('Reducing dimensions...')
        vae = torch.load('model/VAE/VAE_model_for_gcdata.pkl')
        omics_1 = pd.read_csv(path1, sep=',', header=0, index_col=None)
        sample_name = omics_1.iloc[:, 0]
        omics_1 = omics_1.iloc[:, 1:]
        omics_1 = torch.tensor(omics_1.values, dtype=torch.float32).to(device)

        omics_2 = pd.read_csv(path2, sep=',', header=0, index_col=None)
        omics_2 = omics_2.iloc[:, 1:]
        omics_2 = torch.tensor(omics_2.values, dtype=torch.float32).to(device)

        omics_3 = pd.read_csv(path3, sep=',', header=0, index_col=None)
        omics_3 = omics_3.iloc[:, 1:]
        omics_3 = torch.tensor(omics_3.values, dtype=torch.float32).to(device)

        latent_data, latent_data_z, decoded_omics_1, decoded_omics_2, decoded_omics_3, mu_omics1, mu_omics2, mu_omics3, log_var_omics1, log_var_omics2, log_var_omics3 = vae.forward(
            omics_1, omics_2, omics_3)
        latent_df = pd.DataFrame(latent_data.detach().cpu().numpy())
        latent_df.insert(0, 'Sample', sample_name)
        latent_df.to_csv('result/latent_data_gc.csv', header=True, index=False)
        print('Finish!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=int, choices=[0, 1, 2], default=0,
                        help='Mode 0: train&intagrate, Mode information: just train, Mode 2: just intagrate, default: 0.')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed, default=0.')
    parser.add_argument('--num_hidden', '-nh', type=int, default=100, help='The hidden layer dim, default: 100.')
    parser.add_argument('--path1', '-p1', type=str, required=True, help='The first omics file name.')
    parser.add_argument('--path2', '-p2', type=str, required=True, help='The second omics file name.')
    parser.add_argument('--path3', '-p3', type=str, required=True, help='The third omics file name.')
    # parser.add_argument('--pathlabel', '-pl', type=str, required=True, help='The label file name.')
    parser.add_argument('--batchsize', '-bs', type=int, default=64, help='Training batchszie, default: 32.')
    parser.add_argument('--learningrate', '-lr', type=float, default=0.001, help='Learning rate, default: 0.001.')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='Training epochs, default: 100.')
    parser.add_argument('--device', '-d', type=str, choices=['cpu', 'gpu'], default='gpu',
                        help='Training on cpu or gpu, default: cpu.')

    args = parser.parse_args()

    # Check whether GPUs are available
    device = torch.device('cpu')
    if args.device == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set random seed
    setup_seed(args.seed)

    work(epochs=args.epoch, num_hidden=args.num_hidden, mode=args.mode, path1=args.path1, path2=args.path2,
         path3=args.path3, device=device, lr=args.learningrate, batch_size=args.batchsize)
