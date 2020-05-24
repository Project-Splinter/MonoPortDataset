import sys
import os
import json
import time
import tqdm
import torch
from torch.utils.data import DataLoader

from model import HGPIFuNet, ConvPIFuNet
from lib.options import BaseOptions
from lib.dataset import PIFuDataset
from lib.logger import colorlogger

def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

def train(opt):
    # set cuda
    cuda = torch.device('cuda:%d' % opt.gpu_id)

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))
    logger = colorlogger(log_dir=os.path.join(opt.results_path, opt.name))

    train_dataset = PIFuDataset(opt, split='debug')
    test_dataset = PIFuDataset(opt, split='debug')

    projection_mode = train_dataset.projection_mode

    # create data loader
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)

    logger.info(f'train data size: {len(train_data_loader)}')

    # NOTE: batch size should be 1 and use all the points for evaluation
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=1, shuffle=False,
                                  num_workers=4, pin_memory=opt.pin_memory)
    logger.info(f'test data size: {len(test_data_loader)}')

    # create net
    if opt.gtype == "HGPIFuNet":
        netG = HGPIFuNet(opt, projection_mode).to(device=cuda)
    elif opt.gtype == "ConvPIFuNet":
        netG = ConvPIFuNet(opt, projection_mode).to(device=cuda)
    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=opt.learning_rate, momentum=0, weight_decay=0)
    lr = opt.learning_rate
    logger.info(f'Using Network: {netG.name}')
    
    def set_train():
        netG.train()

    def set_eval():
        netG.eval()

    # load checkpoints
    if opt.load_netG_checkpoint_path is not None:
        logger.info(f'loading for net G ... {opt.load_netG_checkpoint_path}')
        netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

    if opt.continue_train:
        if opt.resume_epoch < 0:
            model_path = '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name)
        else:
            model_path = '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, opt.resume_epoch)
        logger.info(f'Resuming from {model_path}')
        netG.load_state_dict(torch.load(model_path, map_location=cuda))

    # training
    start_epoch = 0 if not opt.continue_train else max(opt.resume_epoch,0)
    for epoch in range(start_epoch, opt.num_epoch):
        epoch_start_time = time.time()

        set_train()
        iter_data_time = time.time()
        for train_idx, train_data in enumerate(train_data_loader):
            iter_start_time = time.time()

            # retrieve the data
            image_tensor = train_data['image'].to(device=cuda).float()
            calib_tensor = train_data['calib'].to(device=cuda).float()
            sample_tensor = train_data['samples_geo'].to(device=cuda).float()
            label_tensor = train_data['labels_geo'].to(device=cuda).float()

            sample_tensor = sample_tensor.permute(0, 2, 1) #[bz, 3, N]
            label_tensor = label_tensor.unsqueeze(1) #[bz, 1, N]

            res, error = netG.forward(image_tensor, sample_tensor, calib_tensor, labels=label_tensor)

            optimizerG.zero_grad()
            error.backward()
            optimizerG.step()

            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader) - (
                    iter_net_time - epoch_start_time)

            if train_idx % opt.freq_plot == 0:
                logger.info(
                    'Name: {0} | Epoch: {1} | {2}/{3} | Err: {4:.06f} | LR: {5:.06f} | Sigma: {6:.02f} | dataT: {7:.05f} | netT: {8:.05f} | ETA: {9:02d}:{10:02d}'.format(
                        opt.name, epoch, train_idx, len(train_data_loader), error.item(), lr, opt.sigma_geo,
                                                                            iter_start_time - iter_data_time,
                                                                            iter_net_time - iter_start_time, int(eta // 60),
                        int(eta - 60 * (eta // 60))))

            if train_idx % opt.freq_save == 0 and train_idx != 0:
                torch.save(netG.state_dict(), '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name))
                torch.save(netG.state_dict(), '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, epoch))

            iter_data_time = time.time()

        # update learning rate
        lr = adjust_learning_rate(optimizerG, epoch, lr, opt.schedule, opt.gamma)


if __name__ == '__main__':
    # load args
    parser = BaseOptions().get_parser()
    opt = parser.parse_args()

    # start training
    train(opt)