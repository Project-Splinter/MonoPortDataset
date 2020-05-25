import sys
import os
import json
import time
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import HGPIFuNet, ConvPIFuNet
from lib.options import BaseOptions
from lib.dataset import PIFuDataset
from lib.logger import colorlogger

def update_ckpt(filename, opt, netG, optimizerG, schedulerG, **kwargs):
    ## `kwargs` can be used to store loss, accuracy, epoch, iteration and so on.
    saved_dict = {
        "opt": opt,
        "netG": netG.state_dict(),
        "optimizerG": optimizerG.state_dict(),
        "schedulerG": schedulerG.state_dict(),
    }
    for k, v in kwargs.items():
        saved_dict[k] = v
    torch.save(saved_dict, filename)

def train(opt):
    device = "cuda"

    # set cache path
    checkpoints_path = os.path.join(opt.checkpoints_path, opt.name)
    os.makedirs(checkpoints_path, exist_ok=True)
    results_path = os.path.join(opt.results_path, opt.name)
    os.makedirs(results_path, exist_ok=True)

    # set logger
    logger = colorlogger(log_dir=results_path)
    
    # set dataset
    train_dataset = PIFuDataset(opt, split='debug')
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size, shuffle=not opt.serial_batches,
        num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    logger.info(
        f'train data size: {len(train_dataset)}; '+
        f'loader size: {len(train_data_loader)};')
    
    test_dataset = PIFuDataset(opt, split='debug')
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=1, shuffle=False,
        num_workers=4, pin_memory=opt.pin_memory)
    logger.info(
        f'train data size: {len(test_dataset)}; '+
        f'loader size: {len(test_data_loader)};')

    # set network
    projection_mode = train_dataset.projection_mode
    if opt.gtype == "HGPIFuNet":
        netG = HGPIFuNet(opt, projection_mode).to(device)
    elif opt.gtype == "ConvPIFuNet":
        netG = ConvPIFuNet(opt, projection_mode).to(device)
    else:
        raise NotImplementedError
    netG = nn.DataParallel(netG)
    
    # set optimizer
    learning_rate = opt.learning_rate
    weight_decay = opt.weight_decay
    momentum = opt.momentum
    if opt.optim == "Adadelta":
        optimizerG = torch.optim.Adadelta(
            netG.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif opt.optim == "SGD":
        optimizerG = torch.optim.SGD(
            netG.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif opt.optim == "Adam":
        optimizerG = torch.optim.Adam(
            netG.parameters(), lr=learning_rate)
    elif opt.optim == "RMSprop":
        optimizerG = torch.optim.RMSprop(
            netG.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    else:
        raise NotImplementedError

    # set scheduler
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(
        optimizerG, milestones=opt.schedule, gamma=opt.gamma)

    # ======================== load checkpoints ================================
    ckpt_path = opt.load_netG_checkpoint_path
    if ckpt_path is not None and os.path.exists(ckpt_path):
        logger.info(f"load ckpt from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
    else:
        logger.info(f"ckpt not found: {ckpt_path}")
        ckpt = {}

    # load netG
    if "netG" in ckpt and ckpt["netG"] is not None:
        logger.info('loading for net G ...')
        netG.module.load_state_dict(ckpt["netG"])

    # resume optimizerG & schedulerG
    if opt.continue_train and "optimizerG" in ckpt and ckpt["optimizerG"] is not None:
        logger.info('loading for optimizer G ...')
        optimizerG.load_state_dict(ckpt["optimizerG"])
    if opt.continue_train and "schedulerG" in ckpt and ckpt["schedulerG"] is not None:
        logger.info('loading for scheduler G ...')
        schedulerG.load_state_dict(ckpt["schedulerG"])

    # resume training schedule
    start_epoch = 0
    start_iteration = 0
    if opt.continue_train and "epoch" in ckpt and ckpt["epoch"] is not None:
        start_epoch = ckpt["epoch"]
        logger.info(f'loading for start epoch ... {start_epoch}')
    if opt.continue_train and "iteration" in ckpt and ckpt["iteration"] is not None:
        start_iteration = ckpt["iteration"] + 1
        logger.info(f'loading for start iteration ... {start_iteration}')

    # ==========================================================================

    # start training
    for epoch in range(start_epoch, opt.num_epoch):
        netG.train()
        
        loader = iter(train_data_loader)
        epoch_start_time = iter_start_time = time.time()
        for train_idx in range(start_iteration, len(train_data_loader)):
            train_data = next(loader)            
            iter_data_time = time.time()

            # retrieve the data
            image_tensor = train_data['image'].to(device).float()
            calib_tensor = train_data['calib'].to(device).float()
            sample_tensor = train_data['samples_geo'].to(device).float()
            label_tensor = train_data['labels_geo'].to(device).float()

            sample_tensor = sample_tensor.permute(0, 2, 1) #[bz, 3, N]
            label_tensor = label_tensor.unsqueeze(1) #[bz, 1, N]

            preds, error = netG(image_tensor, sample_tensor, calib_tensor, labels=label_tensor)
            error = error.mean()

            optimizerG.zero_grad()
            error.backward()
            optimizerG.step()

            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (train_idx + 1 - start_iteration)) * len(train_data_loader) - (
                    iter_net_time - epoch_start_time)

            # plot
            if train_idx % opt.freq_plot == 0:
                logger.info(
                    f'Name: {opt.name}|Epoch: {epoch:02d}({train_idx:05d}/{len(train_data_loader)})|' \
                    +f'LR: {schedulerG.get_last_lr()[0]:.4f}|' \
                    +f'dataT: {(iter_data_time - iter_start_time):.3f}|' \
                    +f'netT: {(iter_net_time - iter_data_time):.3f}|'
                    +f'ETA: {int(eta // 60):02d}:{int(eta - 60 * (eta // 60)):02d}|' \
                    +f'Err:{error.item():.5f}|'
                )

            # save
            if train_idx % opt.freq_save == 0 and train_idx != 0:
                update_ckpt(f'{checkpoints_path}/netG_latest', 
                    opt, netG.module, optimizerG, schedulerG, epoch=epoch, iteration=train_idx)
                update_ckpt(f'{checkpoints_path}/netG_epoch_{epoch}', 
                    opt, netG.module, optimizerG, schedulerG, epoch=epoch, iteration=train_idx)
            
            # end
            iter_start_time = time.time()
        
        # end of this epoch
        update_ckpt(f'{checkpoints_path}/netG_epoch_{epoch}', 
                    opt, netG.module, optimizerG, schedulerG, epoch=epoch+1, iteration=-1)
        schedulerG.step()
        start_iteration = 0


if __name__ == '__main__':
    # load args
    parser = BaseOptions().get_parser()
    opt = parser.parse_args()

    # start training
    train(opt)