import os
import sys
sys.path.append('../..')
import numpy as np
import torch
import torch.nn as nn
import random
import logging
from utils import *
import torch.optim as optim
import math
from models.segformer import SegFormer
from train_baselines import train, test
from data_utils import get_data
import yaml


if __name__ == '__main__':

    device = sys.argv[1]
    center_num = (sys.argv[2])
    only_test = (sys.argv[3]=='True')
    fed_learning = (sys.argv[4]=='True')
    config_file = sys.argv[5]
    if fed_learning or sys.argv[6]:
        num_epochs = int(sys.argv[6])
    else:
        num_epochs=2000
    load_path =sys.argv[7]
    
    model = SegFormer(n_channels=64, n_classes=32, config_path='../../mmsegmentation/configs/segformer/segformer_blackfed_camvid.py', device=device)
    if fed_learning:
        model.load_state_dict(torch.load('./fed_learning_model.pth'))

    # with open('data_configs/isic.yml', 'r') as f:
    # with open('data_configs/polypgen.yml', 'r') as f:
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if center_num=='super':
        dataset_dict = get_data(config, center_num='super')
    else:
        dataset_dict = get_data(config, center_num=int(center_num))

    #only train
    if not only_test:
        # model = train(model, dataset_dict, save_path = './baselines/polyp/polyp_baseline_'+str(center_num)+'/', loss_string='bce + dice', device=device, num_epochs=num_epochs)
        # model = test(model, dataset_dict, load_path = './baselines/polyp/polyp_baseline_'+str(center_num)+'/model_best_val.pth', loss_string='bce + dice', device=device)
        model = train(model, dataset_dict, save_path = './saved_models_dice' , loss_string='dice', device=device, num_epochs=num_epochs, center_num=center_num, bs=4, lr=1e-4)
        model = test(model, dataset_dict, load_path = './saved_models_dice/client_'+'best_val.pth', loss_string='dice', device=device)
    else:
        #only test - give load path for model instead of save path
        # model = test(model, dataset_dict, load_path = './skin_baseline_'+str(center_num)+'/model_best_val.pth', loss_string='bce + dice', device=device)
        # model = test(model, dataset_dict, load_path = './baselines/polyp/polyp_baseline_'+str(2)+'/model_best_val.pth', loss_string='bce + dice', device=device)
        # model = test(model, dataset_dict, load_path = './saved_models/client_'+str(int(center_num)-1)+'_best_val.pth', loss_string='bce + dice', device=device)
        model = test(model, dataset_dict, load_path = load_path, loss_string='dice', device=device, center_num=center_num)


