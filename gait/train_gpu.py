from gait.log import get_logger, LoggerWriter
from gait.models import CNN
from gait.gait_dataset import GaitDataset
from gait.models import CNN
from gait.read_data import ReadData
from gait.gait_dataloader import GaitDataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold


logger = get_logger(__name__)
# sys.stdout=LoggerWriter(logger.info)
# sys.stderr=LoggerWriter(logger.error)

def train_func(model, gait_loader, epoch, optimizer, loss_func, FLAGS):
    losses = []
    accuracies = []
    
    scaler = torch.cuda.amp.GradScaler()

    for step, (acc, gyr, mag, targets) in enumerate(gait_loader):
        # training
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(acc, gyr, mag)
            assert output.dtype is torch.float16
            loss = loss_func(output, targets)
            assert loss.dtype is torch.float32
            losses.append(loss.item())

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        # evaluation
        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(acc, gyr, mag)
            pred = torch.max(output, dim = 1)[1]
            acc = (pred == targets).float().mean()
            accuracies.append(acc.item())
        
        print(f'epoch[{epoch}], step {step}/{FLAGS["EPOCHS"]//FLAGS["BATCH_SIZE"]}: acc = {np.mean(accuracies)}, loss = {np.mean(losses)}', end='\r')
    print('\b', end=', ')


def eval_func(model, gait_loader, epoch, optimizer, loss_func, FLAGS):
    model.eval()
    losses = []
    accuracies = []
    
    for acc, gyr, mag, targets in gait_loader:
        with torch.no_grad():
            output = model(acc, gyr, mag)
            loss = loss_func(output, targets)
            losses.append(loss.item())
            pred = torch.max(output, dim = 1)[1]
            acc = (pred == targets).float().mean()
            accuracies.append(acc.item())
        
        print(F'val_acc = {np.mean(accuracies)}, val_loss = {np.mean(losses)}')


def kfold_run(FLAGS, config_dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # read data, data loader
    acc, gyr, mag, targets = ReadData().load_processed_data(config_dict['PREPROCESSED_ARR'])
    
    acc = torch.tensor(acc, dtype=torch.float32, device=device)
    gyr = torch.tensor(gyr, dtype=torch.float32, device=device)
    mag = torch.tensor(mag, dtype=torch.float32, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    kf = KFold(n_splits=5)
    for fold, (train_idx, eval_idx) in enumerate(kf.split(targets)):
        print(f'\nfold: {fold+1}\n')

        train_loader = GaitDataLoader(
            acc[train_idx], gyr[train_idx], mag[train_idx], targets[train_idx],
            batch_size=FLAGS['BATCH_SIZE'],
            shuffle=True,
            drop_last=True
        )

        eval_loader = GaitDataLoader(
            acc[eval_idx], gyr[eval_idx], mag[eval_idx], targets[eval_idx],
            batch_size=eval_idx.shape[0],
            shuffle=False
        )

        #init model, optimizer, loss_func, scheduler
        model = CNN().to(device=device)

        optimizer = optim.Adam(model.parameters(), lr = FLAGS['LEARNING_RATE'])
        loss_func = nn.CrossEntropyLoss()
        # if FLAGS['SCHEDULER']:
        #     scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.9**epoch)

        

        for epoch in range(FLAGS['EPOCHS']):
            # training
            train_func(model, train_loader, epoch, optimizer, loss_func, FLAGS)
            eval_func(model, eval_loader, epoch, optimizer, loss_func, FLAGS)

