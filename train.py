import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import Udacity
from model import Net

import time

def train(model, device, train_loader, optimizer):
    model.train()
    loss_epoch = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output.view(-1).double(), target)
        loss.backward()
        loss_epoch += float(loss.item())
        optimizer.step()
    return loss_epoch / batch_idx


def val(model, device, val_loader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.mse_loss(output.view(-1).double(), target, reduction='sum').item()  # sum up batch loss

    val_loss /= len(val_loader.dataset)
    return val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir-path', type=str, default='data_default', help='data directory path')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    print("Reading csv")
    df = pd.read_csv(str(Path(args.data_dir_path, 'driving_log.csv')), skipinitialspace=True)

    train_df = df.sample(frac=0.8)
    val_df = df.drop(train_df.index)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    train_data = Udacity(df_data=train_df, data_path=args.data_dir_path, transform=transform)
    val_data = Udacity(df_data=val_df, data_path=args.data_dir_path, transform=transform)

    train_loader = DataLoader(dataset=train_data, shuffle=True, **train_kwargs)
    val_loader = DataLoader(dataset=val_data, shuffle=False, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_losses, val_losses = [], []
    start_time = time.time()
    for epoch in range(args.epochs):
        train_loss = train(model, device, train_loader, optimizer)
        train_losses.append(train_loss)
        val_loss = val(model, device, val_loader)
        val_losses.append(val_loss)
        print("Timer {:.2f}".format(time.time()-start_time))
        print('Epoch {} \t Train loss: {:.6f} \t Val loss: {:.6f}'.format(epoch, train_loss, val_loss))

    if args.save_model:
        torch.save(model.state_dict(), "model.pt")

    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training loss', 'validation loss'], loc='upper right')
    plt.savefig('model.png')


if __name__ == '__main__':
    main()