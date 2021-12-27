import time

import torch
import torch.nn as nn

from gait.gait_dataset import GaitDataset
from gait.log import get_logger
from gait.models import CNN
from gait.read_data import ReadData
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

logger = get_logger(__name__)


def run(FLAGS):
    logger.debug('Inside run')

    data = ReadData()._init_data(FLAGS)

    def train_gait(rank, FLAGS):
        # get dataset
        gait_ds = GaitDataset(*data)
        xm.master_print(f'size of gait_ds: {len(gait_ds)}')

        # get sampler
        logger.debug('[xla:{}] Initialize sampler'.format(xm.get_ordinal()))
        gait_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=gait_ds,
            num_replicas=FLAGS['WORLD_SIZE'],
            rank=rank,
            shuffle=True)

        # get dataloader
        logger.debug('[xla:{}] Initialize Loader'.format(xm.get_ordinal()))
        gait_loader = torch.utils.data.DataLoader(
            dataset=gait_ds,
            batch_size=FLAGS['BATCH_SIZE'],
            sampler=gait_sampler,
            num_workers=0,
            drop_last=True)

        # define learning rate
        lr = FLAGS['LEARNING_RATE'] * xm.xrt_world_size()

        # get device
        device = xm.xla_device()

        # get model
        wrapped_model = xmp.MpModelWrapper(CNN())
        model = wrapped_model.to(device=device)
        xm.master_print(model)

        # define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # define loss func
        loss_fn = nn.NLLLoss()

        # loss_fn = nn.CrossEntropyLoss()

        # train func
        def train_func(loader):
            # tracker = xm.RateTracker() ##
            # set to train mode
            model.train()
            for batch_idx, (acc, gyr, mag, targets) in enumerate(loader):
                # setting gradients to zero
                model.zero_grad()

                # calculating output of minibatch
                output = model(acc, gyr, mag)

                # calculating loss of minibatch
                loss = loss_fn(output, targets)

                # backpropagating loss
                loss.backward()

                # running the optimizer step
                xm.optimizer_step(optimizer)

                # printing progress
                if batch_idx % FLAGS['LOG_STEPS'] == 0:
                    print('[xla:{}]({}) Loss={:.5f}'.format(
                        rank, batch_idx, loss.item()))
            # reset to eval model
            model.eval()

        # eval func
        def eval_func(loader):
            model.eval()
            correct = 0
            total_samples = 0
            with torch.no_grad():
                accuracy = 0.0
                for batch_idx, (acc, gyr, mag, targets) in enumerate(loader):
                    output = model(acc, gyr, mag)
                    pred = torch.max(output, dim=1)[1]

                    correct += (pred == targets).sum()
                    total_samples += len(targets)

                    accuracy = 100.0 * correct / total_samples

                return accuracy

        # training/eval loop
        if FLAGS['MODE'] == 'train':
            for epoch in range(FLAGS['EPOCHS']):
                start_time = time.perf_counter()

                print(f'Loading data for rank: {rank}')
                device_loader = pl.ParallelLoader(
                    gait_loader, [device]).per_device_loader(device)

                # calling train_func function to train for one epoch
                print(f'Starting training for rank: {rank}')
                train_func(device_loader)

                print(f'[xla:{rank}] Completed epoch {epoch+1} \
                            & time taken: {time.perf_counter()-start_time}')

        elif FLAGS['MODE'] == 'eval':
            pass

    logger.info('Started training')

    xmp.spawn(train_gait,
              args=(FLAGS, ),
              nprocs=FLAGS['WORLD_SIZE'],
              start_method='fork')
