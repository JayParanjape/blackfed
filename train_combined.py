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


def train(server, clients, dataset_dicts, save_path, loss_string, device):
    server = server.to(device)
    os.makedirs(save_path, exist_ok=True)    
    #set up logger
    logging.basicConfig(filename=os.path.join(save_path,"training_progress.log"),
                    format='%(message)s',
                    filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #set hyperparameters
    num_epochs_server = 51
    num_epochs_client = 21
    bs = 8
    lr_server = 1e-3
    lr_client = 0.01
    sp_avg = 5
    ck = 0.01
    momentum = 0.9

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
    dataloaders = []
    for i in range(len(dataset_dicts)):
        data_loader_client = {}
        data_loader_client['train'] = (torch.utils.data.DataLoader(dataset_dicts[i]['train'], batch_size = bs, shuffle=True, num_workers=4))
        data_loader_client['val'] = (torch.utils.data.DataLoader(dataset_dicts[i]['val'], batch_size=bs, shuffle=False, num_workers=4))
        dataloaders.append(data_loader_client)
    server_optimizer = optim.AdamW(server.parameters(), lr=float(lr_server))

    for i in range(num_epochs_server):
        # zero the parameter gradients
        server_optimizer.zero_grad()
        
        client_outs = []
        client_labels = []
        for j in range(len(dataloaders)):
            inputs, labels,text_idxs, text = next(iter(dataloaders[j]['train']))
            if len(labels.shape)==3:
                labels = labels.unsqueeze(1)

            inputs = inputs.to(device)
            labels = labels.to(device)
            client = clients[j].to(device)


            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                x1 = client(inputs)
                client_outs.append(x1)
                client_labels.append(labels)
        
        with torch.set_grad_enabled(True):
            #merge client outputs
            client_outs = torch.cat(client_outs).to(device)
            client_labels = torch.cat(client_labels).to(device)

            outputs = server(client_outs)
            loss=loss_fxn.forward(outputs, client_labels)
            # print("Reg loss: ",reg_loss)
            
            # backward + optimize only if in training phase
            loss.backward(retain_graph=False)
            server_optimizer.step()

        with torch.no_grad():
            preds = (outputs>=0.5)
            # preds_all.append(preds.cpu())
            # gold.append(labels.cpu())
            # epoch_dice = dice_coef(preds,labels)
            # if count%100==0:
            #     print('iteration dice: ', epoch_dice)


            # statistics
            dice = dice_coef(preds,client_labels)
                
        if i%5==0:
            # epoch_dice = dice_coef(torch.cat(preds_all,axis=0),torch.cat(gold,axis=0))
            print(f'Training at iteration {i} Train Loss: {loss:.4f} Train Dice: {dice:.4f}') 

            #save server
            if loss < best_tr_loss:
                best_tr_loss = loss
                torch.save(server.state_dict(), os.path.join(save_path,'server_best_tr.pth'))
    
    torch.save(server.state_dict(), './tmp_server.pth')

    #train client
    with torch.no_grad():
        for i in range(len(clients)):
            client = clients[i]
            for epoch in range(num_epochs_client):
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
                dataloaders['train'] = torch.utils.data.DataLoader(dataset_dicts[i]['train'], batch_size=bs, shuffle=True, num_workers=4)
                dataloaders['val'] = torch.utils.data.DataLoader(dataset_dicts[i]['val'], batch_size=bs, shuffle=False, num_workers=4)

                for inputs, labels,text_idxs, text in dataloaders['train']:
                # for inputs, labels,text_idxs, text, pt, pt_label in dataloaders[phase]:
                    assert (inputs.shape[0]==labels.shape[0])
                    if len(labels.shape)==3:
                        labels = labels.unsqueeze(1)
                    count+=1
                    intermediate_count += inputs.shape[0]

                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    w = torch.nn.utils.parameters_to_vector(client.parameters())
                    N_params = w.shape[0]

                    ghats = []
                    for spk in range(sp_avg):
                    #! Segmented Uniform [-1, 0.5] U [0.5, 1]
                        p_side = (torch.rand(N_params).reshape(-1,1) + 1)/2
                        samples = torch.cat([p_side,-p_side], dim=1)
                        perturb = torch.gather(samples, 1, torch.bernoulli(torch.ones_like(p_side)/2).type(torch.int64)).reshape(-1).to(w.device)
                        
                        del samples; del p_side

                        #* two-side Approximated Numerical Gradient
                        w_r = w + ck*perturb
                        w_l = w - ck*perturb

                        torch.nn.utils.vector_to_parameters(w_r, client.parameters())
                        x1 = client(inputs)
                        outputs = server(x1)
                        loss_right = loss_fxn.forward(outputs, labels)

                        torch.nn.utils.vector_to_parameters(w_l, client.parameters())
                        x2 = client(inputs)
                        outputs = server(x2)
                        loss_left = loss_fxn.forward(outputs, labels)

                        ghat = (loss_right - loss_left)/((2*ck)*perturb)
                        ghats.append(ghat)
                    torch.cat(ghats, dim=0).mean(dim=0)
                    if count==1:
                        m = ghat
                    else:
                        m = momentum*m + ghat
                    accum_ghat = ghat + momentum*m
                    w = w - lr_client*accum_ghat
                    torch.nn.utils.vector_to_parameters(w, client.parameters())

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

                        # zero the parameter gradients
                        server_optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.no_grad():
                            x1 = client(inputs)
                            outputs= server(x1)
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
                    epoch_dice = running_dice / ((len(dataset_dicts[i]['val'])))
                    # epoch_dice = dice_coef(torch.cat(preds_all,axis=0),torch.cat(gold,axis=0))
                    print(f'Training client at epoch {epoch} Training Loss: {epoch_loss:.4f} Training Dice: {epoch_dice:.4f} HD95 avg: {torch.mean(torch.Tensor(hds))}')

                    #save client
                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        torch.save(client.state_dict(), os.path.join(save_path,'client_'+str(i)+'_best_val.pth'))
    
            torch.save(client.state_dict(), './tmp_client_'+str(i)+'.pth')

    # return server, clients

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

