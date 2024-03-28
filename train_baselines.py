import os
import sys
import numpy as np
import torch
import torch.nn as nn
import random
import logging
from utils import *
import torch.optim as optim
import math
from model import UNet
from data_utils import get_data
import yaml

class Loss_fxn():
    def __init__(self, losses_list=[]):
        if losses_list==[]:
            self.losses_list = [torch.nn.CrossEntropyLoss()]
        else:
            self.losses_list = losses_list

    def forward(self, pred, label):
        tmp_wt = [1,1]
        loss = 0
        for i,l in enumerate(self.losses_list):
            try:
                loss += (tmp_wt[i]*l(pred, label))
            except:
                loss += (tmp_wt[i]*l(pred, label.float()))
        return loss


def train(model, dataset_dict, save_path, loss_string, device, do_only_test=False):
    model = model.to(device)
    os.makedirs(save_path, exist_ok=True)    
    #set up logger
    logging.basicConfig(filename=os.path.join(save_path,"training_progress.log"),
                    format='%(message)s',
                    filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #define loss function
    losses_list = []
    if 'focal' in loss_string:
        losses_list.append(focal_loss)
    if 'dice' in loss_string:
        losses_list.append(dice_loss)
    if 'bce' in loss_string:
        losses_list.append(nn.BCELoss())
    loss_fxn = Loss_fxn(losses_list)
    print("debug: loss loaded")

    if not do_only_test:

        #set hyperparameters
        num_epochs = 2000
        bs = 8
        lr = 1e-3

        # tr_dataloader, val_dataloader = iter(dataloader_dict['train']), iter(dataloader_dict['val'])
        tr_dataset, val_dataset, test_dataset = dataset_dict['train'], dataset_dict['val'], dataset_dict['test']
        best_val_loss = 10000
        best_tr_loss = 10000
        # print("debug: len tr dataset", len(tr_dataset))

        #train model
        best_val_loss = 100000000
        best_tr_loss = 1000000000
        for epoch in range(num_epochs):
            running_loss = 0.0
            running_intersection = 0
            running_union = 0
            running_corrects = 0
            running_dice = 0
            intermediate_count = 0
            count = 0
            preds_all = []
            gold = []
            hds = []
            dataloaders = {}
            dataloaders['train'] = torch.utils.data.DataLoader(dataset_dict['train'], batch_size=bs, shuffle=True, num_workers=4)
            dataloaders['val'] = torch.utils.data.DataLoader(dataset_dict['val'], batch_size=bs, shuffle=False, num_workers=4)
            optimizer = optim.AdamW(model.parameters(), lr=float(lr))

            for inputs, labels,text_idxs, text in dataloaders['train']:
                if len(labels.shape)==3:
                    labels = labels.unsqueeze(1)

                count+=1
                intermediate_count += inputs.shape[0]

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    loss=loss_fxn.forward(outputs, labels)
                    # print("Reg loss: ",reg_loss)
                    
                    # backward + optimize only if in training phase
                    loss.backward(retain_graph=False)
                    optimizer.step()

                with torch.no_grad():
                    preds = (outputs>=0.5)
                    # preds_all.append(preds.cpu())
                    # gold.append(labels.cpu())
                    # epoch_dice = dice_coef(preds,labels)
                    # if count%100==0:
                    #     print('iteration dice: ', epoch_dice)


                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    ri, ru = running_stats(labels,preds)
                    running_dice += dice_collated(ri,ru)
                    
            if epoch%5==0:
                # print(count)
                # print(running_loss, intermediate_count)
                # print(running_loss/intermediate_count)
                epoch_loss = running_loss / ((count))
                epoch_dice = running_dice / ((len(tr_dataset)))
                # epoch_dice = dice_coef(torch.cat(preds_all,axis=0),torch.cat(gold,axis=0))
                print(f'Training at epoch {epoch} Train Loss: {epoch_loss:.4f} Train Dice: {epoch_dice:.4f}') 

            if epoch%10==0:
                running_loss = 0.0
                running_dice = 0
                intermediate_count = 0
                count = 0
                hds = []
                for inputs, labels,text_idxs, text in dataloaders['val']:
                # for inputs, labels,text_idxs, text, pt, pt_label in dataloaders[phase]:
                    if len(labels.shape)==3:
                        labels = labels.unsqueeze(1)
                    count+=1
                    intermediate_count += inputs.shape[0]

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # forward
                    # track history if only in train
                    with torch.no_grad():
                        outputs = model(inputs)
                        loss=loss_fxn.forward(outputs, labels)
                        # print("Reg loss: ",reg_loss)
                        
                        preds = (outputs>=0.5)


                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        ri, ru = running_stats(labels,preds)
                        running_dice += dice_collated(ri,ru)
                        hd = compute_hd95(preds, labels)
                        if not math.isnan(hd):
                            hds.append(hd)
        
                #validation performance
                # print("HD95 avg: ", torch.mean(torch.Tensor(hds)))
                # print(running_loss, intermediate_count)
                # print(running_loss/intermediate_count)
                epoch_loss = running_loss / ((count))
                epoch_dice = running_dice / ((len(tr_dataset)))
                # epoch_dice = dice_coef(torch.cat(preds_all,axis=0),torch.cat(gold,axis=0))
                print(f'Training model at epoch {epoch} Validation Loss: {epoch_loss:.4f} Validation Dice: {epoch_dice:.4f} HD95 avg: {torch.mean(torch.Tensor(hds))}') 
                
                #save model
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    torch.save(model.state_dict(), os.path.join(save_path,'model_best_val.pth'))
    
    else:
        model.load_state_dict(torch.load(os.path.join(save_path,'model_best_val.pth')), strict=True)
    
    #test on test dataset
    running_loss = 0.0
    running_dice = 0
    intermediate_count = 0
    count = 0
    hds = []
    bs=8
    dataloader = torch.utils.data.DataLoader(dataset_dict['test'], batch_size=bs, shuffle=True, num_workers=4)

    for inputs, labels,text_idxs, text in dataloader:
    # for inputs, labels,text_idxs, text, pt, pt_label in dataloaders[phase]:
        if len(labels.shape)==3:
            labels = labels.unsqueeze(1)
        count+=1
        intermediate_count += inputs.shape[0]

        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.no_grad():
            outputs = model(inputs)
            loss=loss_fxn.forward(outputs, labels)
            # print("Reg loss: ",reg_loss)
            
            preds = (outputs>=0.5)


            # statistics
            running_loss += loss.item() * inputs.size(0)
            ri, ru = running_stats(labels,preds)
            running_dice += dice_collated(ri,ru)
            hd = compute_hd95(preds, labels)
            if not math.isnan(hd):
                hds.append(hd)

    epoch_loss = running_loss / ((count))
    epoch_dice = running_dice / ((len(dataset_dict['test'])))
    # epoch_dice = dice_coef(torch.cat(preds_all,axis=0),torch.cat(gold,axis=0))
    print(f'Testing model at epoch {epoch} Test Loss: {epoch_loss:.4f} Test Dice: {epoch_dice:.4f} HD95 avg: {torch.mean(torch.Tensor(hds))}') 

    return model


if __name__ == '__main__':
    model = UNet(n_channels=3, n_classes=1)
    with open('data_configs/isic.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if sys.argv[2]=='super':
        dataset_dict = get_data(config, center_num='super')
    else:
        dataset_dict = get_data(config, center_num=int(sys.argv[2]))

    model = train(model, dataset_dict, save_path = './baseline_'+str(sys.argv[2])+'/', loss_string='bce + dice', device=sys.argv[1], do_only_test=True)