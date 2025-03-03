import os
import argparse
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.autograd import Variable
from configs import Config
from model import get_model
from dataset import get_train_data, get_eval_data
from optim import get_opt_schedule, set_lr
from Loss import get_loss
from utils import set_log, set_seed,run_kmeans
import time
import pandas as pd

from eval import evaluate, Monitor,compute_features


def parse_args():
    parser = argparse.ArgumentParser(description='ssvh')
    parser.add_argument('--config', default='configs/AutoSSVH_fcv.py', type = str,
        help='config file path'
    )
    parser.add_argument('--gpu', default = '0', type = str,
        help = 'specify gpu device'
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if not os.path.exists(cfg.file_path):
        os.makedirs(cfg.file_path)
    
    # set logging
    logger = set_log(cfg, 'log.txt')
    logger.info('Self Supervised Video Hashing Training: {}'.format(cfg.model_name))

    # set seed
    set_seed(cfg)
    logger.info('set seed: {}'.format(cfg.seed))

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True


    gpu_id = int(args.gpu)
    torch.cuda.set_device(gpu_id)
    device = "cuda:"+args.gpu
    logger.info('used gpu: {}'.format(args.gpu))

    logger.info('PARAMETER ......')
    logger.info(cfg)

    logger.info('loading model ......') 
    model = get_model(cfg).to(device)
    logger.info("encoder param:{}".format(model.get_encoder_param_count()))

    logger.info('loading train data ......')    
    train_loader = get_train_data(cfg)
    total_len = len(train_loader)

    # load eval data
    logger.info('loading eval data ......')     
    eval_loader = get_eval_data(cfg)

    epoch = 0
    
    # optimizer and schedule
    opt_schedule = get_opt_schedule(cfg, model)
    criterion = nn.CrossEntropyLoss(reduction="mean").to(device)
    if cfg.use_checkpoint is not None:
        checkpoint = torch.load(cfg.use_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt_schedule._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        opt_schedule._schedule.last_epoch = checkpoint['epoch']
    
    monitor = Monitor(max_patience=10, delta=1e-5)

    # mAP5
    mAP5_max=-99
    mAP5_now=-999

    # Obtain clustering results
    while True:
        cluster_result = None

        if  cfg.CVH and epoch >=cfg.warmup_epoch:
            features = compute_features(train_loader, model, cfg,device)         

            # placeholder for clustering result
            cluster_result = {'im2cluster':[],'centroids':[],'density':[]}
            for num_cluster in cfg.num_cluster:
                cluster_result['im2cluster'].append(torch.zeros(cfg.train_num_sample,dtype=torch.long).to(device))#每个特征对应的簇索引
                cluster_result['centroids'].append(torch.zeros(int(num_cluster),cfg.nbits).to(device))
                cluster_result['density'].append(torch.zeros(int(num_cluster)).to(device)) 

            features = features.numpy()
            cluster_result = run_kmeans(features,cfg,gpu_id,device)  #run kmeans clustering on master node
            # save the clustering result
            # torch.save(cluster_result,os.path.join(args.exp_dir, 'clusters_%d'%epoch))    

        logger.info('begin training stage: [{}/{}]'.format(epoch+1, cfg.num_epochs)) 

        # Regular evaluation
        if cfg.dataset == 'fcv':
            if epoch % 10 == 0 and  epoch != 0:
                mAP5_now = evaluate(cfg, model, cfg.test_num_sample ,logger,device,train=True)
        elif cfg.dataset == 'activitynet' or cfg.dataset=="ucf" or cfg.dataset=="hmdb":
            if epoch % 1 == 0 and  epoch != 0:
                mAP5_now = evaluate(cfg, model, cfg.test_num_sample ,logger,device,train=True)

        #  Save best checkpoint -- mAP[5]
        if mAP5_now > mAP5_max:
            save_file = cfg.file_path + '/{}_{}.pth'.format(cfg.dataset, cfg.nbits,eval_loader)
            torch.save({
                'model_state_dict': model.state_dict()
            }, save_file)
            mAP5_max = mAP5_now

        logger.info('begin training stage: [{}/{}]'.format(epoch+1, cfg.num_epochs))  


        model.train()
            
        for i, data in enumerate(train_loader, start=1):
            opt_schedule.zero_grad()
            loss = get_loss(cfg, data, model, epoch, i, total_len, logger,cluster_result,criterion,device)
            loss.backward()
            opt_schedule._optimizer_step()
        
        opt_schedule._schedule_step()
        logger.info('now the learning rate is: {}'.format(opt_schedule.lr()))

        epoch += 1
        if epoch >= cfg.num_epochs:
            break

if __name__ == '__main__':
    main()


