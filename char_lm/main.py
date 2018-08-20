#!/usr/bin/env python3
import numpy as np

import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from char_lm.model import CausalNet
from char_lm.dataset import TextSet

import click

class EarlyStopping(object):
    """
    Early stopping to terminate training when validation loss doesn't improve
    over a certain time.
    """
    def __init__(self, it=None, min_delta=0.002, lag=5):
        """
        Args:
            it (torch.utils.data.DataLoader): training data loader
            min_delta (float): minimum change in validation loss to qualify as improvement.
            lag (int): Number of epochs to wait for improvement before
                       terminating.
        """
        self.min_delta = min_delta
        self.lag = lag
        self.it = it
        self.best_loss = 0
        self.wait = 0

    def __iter__(self):
        return self

    def __next__(self):
        #if self.wait >= self.lag:
        #     raise StopIteration
        return self.it

    def update(self, val_loss):
        """
        Updates the internal validation loss state
        """
        if (val_loss - self.best_loss) < self.min_delta:
            self.wait += 1
        else:
            self.wait = 0
            self.best_loss = val_loss

@click.group()
def cli():
    pass

@cli.command()
@click.option('-n', '--name', default='model', help='prefix for checkpoint file names')
@click.option('-l', '--lrate', default=0.1, help='initial learning rate')
@click.option('-w', '--workers', default=0, help='number of workers loading training data')
@click.option('-d', '--device', default='cpu', help='pytorch device')
@click.option('-v', '--validation', default='val', help='validation set location')
@click.option('--lag', show_default=True, default=20, help='Number of epochs to wait before stopping training without improvement')
@click.option('--min-delta', show_default=True, default=0.005, help='Minimum improvement between epochs to reset early stopping')
@click.option('--optimizer', show_default=True, default='SGD', type=click.Choice(['SGD', 'Adam']), help='optimizer')
@click.option('--threads', default=min(len(os.sched_getaffinity(0)), 4))
@click.option('--valid-seq-len', default=320, help='part of the training sample used for back propagation')
@click.option('--seq-len', default=400, help='total training sample sequence length')
@click.option('--hidden', default=100, help='numer of hidden units per block')
@click.option('--layers', default=2, help='number of 3-layer blocks')
@click.option('--kernel', default=3, help='kernel size')
@click.argument('ground_truth', nargs=1)
def train(name, lrate, workers, device, validation, lag, min_delta, optimizer,
          threads, valid_seq_len, seq_len, hidden, layers, kernel,
          ground_truth):

    print('model output name: {}'.format(name))

    torch.set_num_threads(threads)

    train_set = TextSet(glob.glob('{}/**/*.txt'.format(ground_truth), recursive=True))
    train_data_loader = DataLoader(dataset=train_set, num_workers=workers, batch_size=16, pin_memory=True)
    val_set = TextSet(glob.glob('{}/**/*.txt'.format(validation), recursive=True), chars=train_set.chars)
    val_data_loader = DataLoader(dataset=val_set, num_workers=workers, batch_size=16, pin_memory=True)

    device = torch.device(device)

    print('loading network')

    model = CausalNet(len(train_set.chars), len(train_set.chars), hidden, layers, kernel).to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer == 'SGD':
        opti = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate)
    else:
        opti = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate)
    st_it = EarlyStopping(train_data_loader, min_delta, lag)
    val_loss = 9999
    for epoch, loader in enumerate(st_it):
        epoch_loss = 0
        with click.progressbar(train_data_loader, label='epoch {}'.format(epoch)) as bar:
            for sample in bar:
                input, target = sample[0].to(device, non_blocking=True), sample[1].to(device, non_blocking=True)
                opti.zero_grad()
                o = model(input)
                o = o[:, seq_len - valid_seq_len, :].contiguous()
                target = target[:, :seq_len - valid_seq_len].contiguous()
                loss = criterion(o, target)
                epoch_loss += loss.item()
                loss.backward()
                opti.step()
        torch.save(model.state_dict(), '{}_{}.ckpt'.format(name, epoch))
        print("===> epoch {} complete: avg. loss: {:.4f}".format(epoch, epoch_loss / len(train_data_loader)))
        val_loss = evaluate(model, device, criterion, val_data_loader, seq_len, valid_seq_len)
        model.train()
        st_it.update(val_loss)
        print("===> epoch {} validation loss: {:.4f}".format(epoch, val_loss))

def evaluate(model, device, criterion, data_loader, seq_len, valid_seq_len):
    """
    """
    model.eval()
    loss = 0.0
    with torch.no_grad():
         for sample in data_loader:
             input, target = sample[0].to(device), sample[1].to(device)
             o = model(input)
             o = o[:, seq_len - valid_seq_len, :].contiguous()
             target = target[:, seq_len - valid_seq_len, :].contiguous()
             loss += criterion(o, target)
    return loss / len(data_loader)
