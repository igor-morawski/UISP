import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import Myloss
import numpy as np
from torchvision import transforms

from tensorboardX import SummaryWriter
from datetime import datetime


EXPERIMENT_DIR = "experiments"
MODEL_NAME = "ZDCE_unsupervised"
CPT_DIR = "checkpoints"
LOG_DIR = "logs"

def mk_and_assert_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    assert os.path.exists(dir)


def train(config):
    model_dir = os.path.join(EXPERIMENT_DIR, MODEL_NAME+"_"+datetime.today().strftime('%Y%m%d_%H%M'))
    mk_and_assert_dir(model_dir)

    checkpoint_dir = os.path.join(model_dir, CPT_DIR)
    mk_and_assert_dir(checkpoint_dir)
    
    log_dir = os.path.join(model_dir, LOG_DIR)
    mk_and_assert_dir(log_dir)
    
    log_writter = SummaryWriter(log_dir)

    DCE_net = model.enhance_net_nopool(in_channels=4).cuda()

    DCE_net.apply(DCE_net.weights_init)
    train_dataset = dataloader.loader_SID(config.dataset_path, config.camera, 'train', patch_size=config.patch_size, upsample=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    L_color = Myloss.L_color()
    L_spa = Myloss.L_spa()

    L_exp = Myloss.L_exp(16, 0.6)
    L_TV = Myloss.L_TV()

    optimizer = torch.optim.Adam(DCE_net.parameters(
    ), lr=config.lr, weight_decay=config.weight_decay)

    DCE_net.train()

    iteration_idx = 0
    for epoch in range(config.num_epochs):
        for iteration, img_lowlight in enumerate(train_loader):
            iteration_idx += 1
            
            img_lowlight = img_lowlight.cuda()

            enhanced_image_1, enhanced_image, A = DCE_net(img_lowlight)

            Loss_TV = 200*L_TV(A)
            loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))
            loss_col = 5*torch.mean(L_color(enhanced_image))
            loss_exp = 10*torch.mean(L_exp(enhanced_image))

            # best_loss
            loss = Loss_TV + loss_spa + loss_col + loss_exp
            #
            
            log_writter.add_scalar('loss_tv', Loss_TV.item(), iteration_idx)
            log_writter.add_scalar('loss_spa', loss_spa.item(), iteration_idx)
            log_writter.add_scalar('loss_col', loss_col.item(), iteration_idx)
            log_writter.add_scalar('loss_exp', loss_exp.item(), iteration_idx)
            log_writter.add_scalar('loss_total', loss.item(), iteration_idx)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(
                DCE_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            if ((iteration+1) % config.display_iter) == 0:
                print("Loss at iteration", iteration+1, ":", loss.item())

        torch.save(DCE_net.state_dict(), 
                    os.path.join(checkpoint_dir, f"Epoch_{epoch}.pth"))
        torch.save(DCE_net.state_dict(), 
                    os.path.join(checkpoint_dir, f"last.pth"))
            
def test(config):
    checkpoint = config.checkpoint
    results_dir = os.path.join(EXPERIMENT_DIR, "results") 
    mk_and_assert_dir(results_dir)
    model_dir = os.path.join(results_dir, os.path.split(os.path.split(os.path.split(checkpoint)[0])[0])[-1]) 
    mk_and_assert_dir(model_dir)
    print(f"Results will be saved to {model_dir}")
    
    DCE_net = model.enhance_net_nopool(in_channels=4).cuda()
    DCE_net.load_state_dict(torch.load(checkpoint))
    
    test_dataset = dataloader.loader_SID(config.dataset_path, config.camera, 'test', patch_size=config.patch_size, return_gt=True, upsample=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    for iteration, img_lowlight in enumerate(test_loader):
        print(iteration)
        data_lowlight, gt_data, in_fp, gt_fp = img_lowlight
        data_lowlight = data_lowlight.cuda()
        gt_data = gt_data.cuda()
        _,enhanced_image,_ = DCE_net(data_lowlight)
        in_fn, gt_fn = os.path.split(in_fp[0])[-1], os.path.split(gt_fp[0])[-1]
        in_fn, gt_fn = in_fn.replace("ARW","JPG"), gt_fn.replace("ARW","JPG")
        torchvision.utils.save_image(enhanced_image, os.path.join(model_dir, in_fn)) #XXX 0.jpg -> name, gt, no patch
        torchvision.utils.save_image(gt_data, os.path.join(model_dir, gt_fn)) #XXX 0.jpg -> name, gt, no patch
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--dataset_path', type=str, default="data/SID/")
    parser.add_argument('--camera', type=str, default="Sony")
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--checkpoint', type=str)


    config = parser.parse_args()

    if not os.path.exists(EXPERIMENT_DIR):
        os.mkdir(EXPERIMENT_DIR)
    assert os.path.exists(EXPERIMENT_DIR)
    
    if not config.test:
        train(config)
    else:
        test(config)
