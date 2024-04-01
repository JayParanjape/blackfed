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


def train(server, client, dataset_dict, j, save_path, loss_string, device):
    server = server.to(device)
    client = client.to(device)
    os.makedirs(save_path, exist_ok=True)    
    #set up logger
    logging.basicConfig(filename=os.path.join(save_path,"training_progress.log"),
                    format='%(message)s',
                    filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #set hyperparameters
    num_epochs = 21
    bs = 32
    lr_server = 1e-3
    lr_client = 0.001
    sp_avg = 5
    ck = 0.01
    momentum = 0.9

    # tr_dataloader, val_dataloader = iter(dataloader_dict['train']), iter(dataloader_dict['val'])
    tr_dataset, val_dataset = dataset_dict['train'], dataset_dict['val']
    best_val_loss = 10000
    best_tr_loss = 10000
    # print("debug: len tr dataset", len(tr_dataset))


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

    #train server
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
        server_optimizer = optim.AdamW(server.parameters(), lr=float(lr_server))
        client_optimizer = optim.AdamW(client.parameters(), lr=float(lr_client))

        for inputs, labels,text_idxs, text in dataloaders['train']:
            if len(labels.shape)==3:
                labels = labels.unsqueeze(1)
        # for inputs, labels,text_idxs, text, pt, pt_label in dataloaders[phase]:

            count+=1
            intermediate_count += inputs.shape[0]

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            server_optimizer.zero_grad()
            client_optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                x1 = client(inputs)
            with torch.set_grad_enabled(True):
                outputs = server(x1)
                loss=loss_fxn.forward(outputs, labels)
                # print("Reg loss: ",reg_loss)
                
                # backward + optimize only if in training phase
                loss.backward(retain_graph=False)
                server_optimizer.step()
                client_optimizer.step()

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
            print(count)
            print(running_loss, intermediate_count)
            print(running_loss/intermediate_count)
            epoch_loss = running_loss / ((len(tr_dataset)))
            epoch_dice = running_dice / ((len(tr_dataset)))
            # epoch_dice = dice_coef(torch.cat(preds_all,axis=0),torch.cat(gold,axis=0))
            print(f'Training at epoch {epoch} Train Loss: {epoch_loss:.4f} Train Dice: {epoch_dice:.4f}') 

        if epoch%20==0:
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
                    x1 = client(inputs)
                    outputs= server(x1)
                    loss=loss_fxn.forward(outputs, labels)
                    # print("Reg loss: ",reg_loss)
                    
                    preds = (outputs>=0.5)+0

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
            epoch_loss = running_loss / ((len(dataset_dict['val'])))
            epoch_dice = running_dice / ((len(dataset_dict['val'])))
            # epoch_dice = dice_coef(torch.cat(preds_all,axis=0),torch.cat(gold,axis=0))
            print(f'Training server at epoch {epoch} Validation Loss: {epoch_loss:.4f} Validation Dice: {epoch_dice:.4f} HD95 avg: {torch.mean(torch.Tensor(hds))}') 
               
            #save server
            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                torch.save(server.state_dict(), os.path.join(save_path,'server_best_val.pth'))
                torch.save(client.state_dict(), os.path.join(save_path,'client_best_val.pth'))
    
    torch.save(server.state_dict(), './tmp_server.pth')
    torch.save(client.state_dict(), './tmp_client_'+str(j)+'.pth')

    return server, client

def test(server, client, dataset_dict, device):
    server = server.to(device)
    client = client.to(device)
    #set up logger
    # logging.basicConfig(filename=os.path.join('saved_models',"testing_progress.log"),
    #                 format='%(message)s',
    #                 filemode='a')
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    bs = 8
    test_dataloader = torch.utils.data.DataLoader(dataset_dict['test'], batch_size=bs, shuffle=False, num_workers=4)
    hds = []
    count = 0
    running_dice = 0
    for img, label, _,_ in test_dataloader:
        with torch.no_grad():
            if len(label.shape)==3:
                label = label.unsqueeze(1)
            img = img.to(device)
            label = label.to(device)
            count += bs
            x1 = client(img)
            outputs= server(x1)
            preds = (outputs>=0.5)

            # statistics
            ri, ru = running_stats(label,preds)
            running_dice += dice_collated(ri,ru)
            hd = compute_hd95(preds, label)
            if not math.isnan(hd):
                hds.append(hd)

    #validation performance
    # print("HD95 avg: ", torch.mean(torch.Tensor(hds)))
    # print(running_loss, intermediate_count)
    # print(running_loss/intermediate_count)
    epoch_dice = running_dice / len(dataset_dict['test'])
    # epoch_dice = dice_coef(torch.cat(preds_all,axis=0),torch.cat(gold,axis=0))
    print(f'Testing {dataset_dict["name"]}: Dice: {epoch_dice:.4f} HD95 avg: {torch.mean(torch.Tensor(hds))}')

