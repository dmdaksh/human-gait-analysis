from gait.log import get_logger
from gait.models import CNN
from gait.gait_dataset import GaitDataset
from gait.models import CNN
from gait.read_data import ReadData
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold


def train_func(model, gait_loader, epoch, optimizer, loss_func):
    losses = []
    accuracies = []
    pbar = tqdm(enumerate(gait_loader), desc=f'epoch {epoch+1}, steps:')
    
    scaler = torch.cuda.amp.GradScaler()

    for step, (acc, gyr, mag, targets) in pbar:
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

        scaler.step()

        scaler.update()

        # evaluation
        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(acc, gyr, mag)
            pred = torch.max(output, dim = 1)[1]
            acc = (pred == targets).float().mean()
            accuracies.append(acc.item())
        
        pbar.set_postfix(acc = np.mean(accuracies), loss = np.mean(losses))


def eval_func(model, gait_loader, epoch, optimizer, loss_func):
    model.eval()
    losses = []
    accuracies = []
    pbar = tqdm(enumerate(gait_loader), desc=f'epoch {epoch+1}, val_steps:')
    for step, (acc, gyr, mag, targets) in pbar:
        with torch.no_grad():
            output = model(acc, gyr, mag)
            loss = loss_func(output, targets)
            losses.append(loss.item())
            pred = torch.max(output, dim = 1)[1]
            acc = (pred == targets).float().mean()
            accuracies.append(acc.item())
        
        pbar.set_postfix(val_acc = np.mean(accuracies), val_loss = np.mean(losses))


def kfold_run(FLAGS, config_dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # read data, data loader
    acc, gyr, mag, targets = ReadData().load_processed_data(config_dict['PREPROCESSED_ARR'])

    kf = KFold(n_splits=5)
    for train_idx, eval_idx in kf.split(targets):
        gait_ds = GaitDataset(acc=acc[train_idx], gyr=gyr[train_idx], mag=mag[train_idx], targets=targets[train_idx], device=device)

        gait_loader = torch.utils.data.DataLoader(
            dataset=gait_ds,
            batch_size=FLAGS['BATCH_SIZE'],
            drop_last=True,
            shuffle=True
        )

        eval_gait_ds = GaitDataset(acc=acc[eval_idx], gyr=gyr[eval_idx], mag=mag[eval_idx], targets=targets[eval_idx], device=device)

        eval_gait_loader = torch.utils.data.DataLoader(
            dataset=eval_gait_ds,
            batch_size=eval_idx.shape[0]
        )

        #init model, optimizer, loss_func, scheduler
        model = CNN().to(device=device)

        optimizer = optim.Adam(model.parameters(), lr = FLAGS['LEARNING_RATE'])
        loss_func = nn.CrossEntropyLoss()
        # if FLAGS['SCHEDULER']:
        #     scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.9**epoch)

        

        for epoch in range(FLAGS['EPOCHS']):
            # training
            train_func(model, gait_loader, epoch, optimizer, loss_func)
            eval_func(model, eval_gait_loader, epoch, optimizer, loss_func)

