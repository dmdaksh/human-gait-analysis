from dataclasses import asdict
from os import device_encoding

from sklearn.utils import shuffle
from gait.log import get_logger
from gait.config import Config
from gait.models import CNN
from gait.gait_dataset import GaitDataset
from gait.models import CNN
from gait.read_data import ReadData
import torch
import torch.nn as nn
import torch.optim as optim
import time
import gc
import numpy as np

def run(FLAGS):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # read data, data loader
    data = ReadData()._init_data(FLAGS)

    gait_ds = GaitDataset(*data, device=device)

    gait_loader = torch.utils.data.DataLoader(
        dataset=gait_ds,
        batch_size=FLAGS['BATCH_SIZE'],
        drop_last=True,
        shuffle=True
    )

    #init model, optimizer, loss_func, scheduler
    model = CNN().to(device=device)

    optimizer = optim.Adam(model.parameters(), lr = FLAGS['LEARNING_RATE'])
    loss_func = nn.CrossEntropyLoss()
    # if FLAGS['SCHEDULER']:
    #     scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.9**epoch)

    # training
    losses = []
    accuracies = []
    for step, (acc, gyr, mag, targets) in enumerate(gait_loader):
        model.train()

        optimizer.zero_grad()

        output = model(acc, gyr, mag)

        loss = loss_func(output, targets)

        loss.backward()
        losses.append(loss.item())

        optimizer.step()


        model.eval()
        with torch.no_grad():
            output = model(acc, gyr, mag)
            pred = torch.max(output, dim = 1)[1]
            acc = (pred == targets).float().mean()
            accuracies.append(acc)
        
        print(f'step: {step}, acc: {np.mean(accuracies)}, loss: {np.mean(losses)}')


