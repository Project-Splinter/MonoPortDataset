import sys
import os
import json
import time
import tqdm
import numpy as np
from skimage import measure

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from implicit_seg.functional import Seg3dLossless

from model import HGPIFuNet, ConvPIFuNet
from lib.options import BaseOptions
from lib.dataset import PIFuDataset
from lib.logger import colorlogger

device = "cuda"

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

def query_func(opt, netG, features, points, proj_matrix=None):
    '''
        - points: size of (bz, N, 3)
        - proj_matrix: size of (bz, 4, 4)
    return: size of (bz, 1, N)
    '''
    assert len(points) == 1
    samples = points.repeat(opt.num_views, 1, 1)
    samples = samples.permute(0, 2, 1) # [bz, 3, N]

    # view specific query
    if proj_matrix is not None:
        samples = orthogonal(samples, proj_matrix)

    calib_tensor = torch.stack([
        torch.eye(4).float()
    ], dim=0).to(samples.device)

    preds = netG.query(
        features=features,
        points=samples, 
        calibs=calib_tensor)
    if type(preds) is list:
        preds = preds[0]
    return preds
    

def train(
    opt, data_loader, netG, optimizerG, schedulerG,
    logger, tb_writer, reconEngine, checkpoints_path, epoch, start_iter=0):
    netG.train()

    epoch_start_time = iter_start_time = time.time()
    loader = iter(data_loader)
    niter = len(data_loader)
    for iteration in range(start_iter, niter):
        train_data = next(loader)            
        iter_data_time = time.time() - iter_start_time
        global_step = epoch * niter + iteration

        # retrieve the data
        image_tensor = train_data['image'].to(device).float()
        calib_tensor = train_data['calib'].to(device).float()
        sample_tensor = train_data['samples_geo'].to(device).float()
        label_tensor = train_data['labels_geo'].to(device).float()

        sample_tensor = sample_tensor.permute(0, 2, 1) #[bz, 3, N]
        label_tensor = label_tensor.unsqueeze(1) #[bz, 1, N]

        preds, error = netG(
            image_tensor, sample_tensor, calib_tensor, labels=label_tensor)
        error = error.mean()

        optimizerG.zero_grad()
        error.backward()
        optimizerG.step()

        iter_time = time.time() - iter_start_time
        eta = (niter-start_iter) * (time.time()-epoch_start_time) / (iteration-start_iter+1) 

        # print
        if iteration % opt.freq_plot == 0:
            logger.info(
                f'Name: {opt.name}|Epoch: {epoch:02d}({iteration:05d}/{niter})|' \
                +f'dataT: {(iter_data_time):.3f}|' \
                +f'totalT: {(iter_time):.3f}|'
                +f'ETA: {int(eta // 60):02d}:{int(eta - 60 * (eta // 60)):02d}|' \
                +f'Err:{error.item():.5f}|'
            )
            tb_writer.add_scalar('data/errorG', error.item(), global_step)

        # recon
        if iteration % opt.freq_recon == 0:
            logger.info('generate mesh (test) ...')
            netG.eval()            
            features = netG.module.filter(image_tensor[0:1])
            sdf = reconEngine(opt=opt, netG=netG.module, features=features, proj_matrix=None)
            sdf = sdf[0, 0].detach().cpu().numpy()
            
            try:
                verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, 0.5)
                tb_writer.add_mesh(
                    'recon', vertices=verts[np.newaxis], faces=faces[np.newaxis], global_step=global_step)
            except Exception as e:
                print(f'error cannot marching cubes: {e}')
            netG.train()

        # save
        if iteration % opt.freq_save == 0:
            update_ckpt(f'{checkpoints_path}/netG_latest', 
                opt, netG.module, optimizerG, schedulerG, epoch=epoch, iteration=iteration)
            update_ckpt(f'{checkpoints_path}/netG_epoch_{epoch}', 
                opt, netG.module, optimizerG, schedulerG, epoch=epoch, iteration=iteration)
        
        # end
        iter_start_time = time.time()
        

def main(opt):
    # hierachy occupancy reconstruction
    reconEngine = Seg3dLossless(
        query_func=query_func, 
        b_min=[[-1.0,  1.0, -1.0]],
        b_max=[[ 1.0, -1.0,  1.0]],
        resolutions=[16+1, 32+1, 64+1, 128+1],
        align_corners=False,
        balance_value=0.5,
        device=device, 
        visualize=False,
        debug=False,
        use_cuda_impl=True,
        faster=True)

    # set cache path
    checkpoints_path = os.path.join(opt.checkpoints_path, opt.name)
    os.makedirs(checkpoints_path, exist_ok=True)
    results_path = os.path.join(opt.results_path, opt.name)
    os.makedirs(results_path, exist_ok=True)

    # set logger
    logger = colorlogger(log_dir=results_path)

    # set tensorboard
    tb_writer = SummaryWriter(logdir=results_path)
    
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
    start_iter = 0
    if opt.continue_train and "epoch" in ckpt and ckpt["epoch"] is not None:
        start_epoch = ckpt["epoch"]
        logger.info(f'loading for start epoch ... {start_epoch}')
    if opt.continue_train and "iteration" in ckpt and ckpt["iteration"] is not None:
        start_iter = ckpt["iteration"] + 1
        logger.info(f'loading for start iteration ... {start_iteration}')
    # ==========================================================================

    # start training
    for epoch in range(start_epoch, opt.num_epoch):
        netG.train()
        train(
            opt, train_data_loader, netG, optimizerG, schedulerG,
            logger, tb_writer, reconEngine, checkpoints_path, epoch, start_iter)
        schedulerG.step()
        start_iter = 0


if __name__ == '__main__':
    # load args
    parser = BaseOptions().get_parser()
    opt = parser.parse_args()

    # start training
    main(opt)