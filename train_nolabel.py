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


def train_no_label(server, client, dataset_dict, j, save_path, loss_string, device):
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
    num_epochs_server = 31
    num_epochs_client_enc = 31
    num_epochs_client_dec = 11
    bs = 32
    lr_server = 0.005
    lr_client_enc = 0.01
    lr_client_dec = 1e-3
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

    #train client decoder
    best_tr_loss = 100000000
    best_val_loss = 100000000
    for epoch in range(num_epochs_client_dec):
        running_loss = 0.0
        running_dice = 0.0
        hds = []
        dataloaders = {}
        dataloaders['train'] = torch.utils.data.DataLoader(dataset_dict['train'], batch_size=bs, shuffle=True, num_workers=4)
        dataloaders['val'] = torch.utils.data.DataLoader(dataset_dict['val'], batch_size=bs, shuffle=False, num_workers=4)
        client_dec_optimizer = optim.AdamW( [client.decoder_layers, client.outc] , lr=float(lr_client_dec))

        for inputs, labels, _, _ in dataloaders['train']:
            if len(labels.shape)==3:
                labels = labels.unsqueeze(1)
            
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        client_dec_optimizer.zero_grad()

        #forward pass through the black parts assuming constant parameters
        with torch.set_grad_enabled(False):
            x1 = client(inputs, 'encode')
            x2 = server(x1)
        
        #forward pass through the white parts of the client
        with torch.set_grad_enabled(True):
            x3 = client(x2, 'decode')
            loss=loss_fxn.forward(x2, labels)
            loss.backward(retain_graph=False)
            client_dec_optimizer.step()

        with torch.no_grad():
            preds = (x3>=0.5)
            # statistics
            running_loss += loss.item() * inputs.size(0)
            ri, ru = running_stats(labels,preds)
            running_dice += dice_collated(ri,ru)

        if epoch%5==0:
            epoch_loss = running_loss / ((len(tr_dataset)))
            epoch_dice = running_dice / ((len(tr_dataset)))
            print(f'Training client decoder at epoch {epoch} Train Loss: {epoch_loss:.4f} Train Dice: {epoch_dice:.4f}')


    #train server
    with torch.no_grad():
        for epoch in range(num_epochs_server):
            running_loss = 0.0
            running_dice = 0
            hds = []
            dataloaders = {}
            dataloaders['train'] = torch.utils.data.DataLoader(dataset_dict['train'], batch_size=bs, shuffle=True, num_workers=4)
            dataloaders['val'] = torch.utils.data.DataLoader(dataset_dict['val'], batch_size=bs, shuffle=False, num_workers=4)

            for inputs, labels,_, _ in dataloaders['train']:
                assert (inputs.shape[0]==labels.shape[0])
                if len(labels.shape)==3:
                    labels = labels.unsqueeze(1)
                count+=1
                intermediate_count += inputs.shape[0]

                inputs = inputs.to(device)
                labels = labels.to(device)
                w = torch.nn.utils.parameters_to_vector(server.parameters())
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

                    torch.nn.utils.vector_to_parameters(w_r, server.parameters())
                    x1 = client(inputs, 'encode')
                    x2 = server(x1)
                    x3 = client(x2, 'decode')
                    
                    loss_right = loss_fxn.forward(x3, labels)

                    torch.nn.utils.vector_to_parameters(w_l, server.parameters())
                    x1 = client(inputs, 'encode')
                    x2 = server(x1)
                    x3 = client(x2, 'decode')

                    loss_left = loss_fxn.forward(x3, labels)

                    ghat = (loss_right - loss_left)/((2*ck)*perturb)
                    ghats.append(ghat)

                torch.cat(ghats, dim=0).mean(dim=0)
                if count==1:
                    m = ghat
                else:
                    m = momentum*m + ghat
                accum_ghat = ghat + momentum*m
                w = w - lr_server*accum_ghat
                torch.nn.utils.vector_to_parameters(w, server.parameters())

        if epoch%20==0:
            running_loss = 0.0
            running_dice = 0
            hds = []
            for inputs, labels,_,_ in dataloaders['train']:
                if len(labels.shape)==3:
                    labels = labels.unsqueeze(1)

                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.no_grad():
                    x1 = client(inputs, 'encode')
                    x2 = server(x1)
                    x3 = client(x2, 'decode')

                    loss=loss_fxn.forward(x3, labels)
                    # print("Reg loss: ",reg_loss)
                    
                    preds = (x3>=0.5)


                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    ri, ru = running_stats(labels,preds)
                    running_dice += dice_collated(ri,ru)
                    hd = compute_hd95(preds, labels)
                    if not math.isnan(hd):
                        hds.append(hd)
    
            epoch_loss = running_loss / ((len(dataset_dict['train'])))
            epoch_dice = running_dice / ((len(dataset_dict['train'])))
            print(f'Training server at epoch {epoch} Training Loss: {epoch_loss:.4f} Training Dice: {epoch_dice:.4f} HD95 avg: {torch.mean(torch.Tensor(hds))}')

            #save server
            if epoch_loss < best_tr_loss:
                best_tr_loss = epoch_loss
                torch.save(server.state_dict(), os.path.join(save_path,'server_best_tr.pth'))

    #train client encoder
    with torch.no_grad():
        for epoch in range(num_epochs_client_enc):
            running_loss = 0.0
            running_dice = 0
            hds = []
            dataloaders = {}
            dataloaders['train'] = torch.utils.data.DataLoader(dataset_dict['train'], batch_size=bs, shuffle=True, num_workers=4)
            dataloaders['val'] = torch.utils.data.DataLoader(dataset_dict['val'], batch_size=bs, shuffle=False, num_workers=4)

            for inputs, labels,_, _ in dataloaders['train']:
                assert (inputs.shape[0]==labels.shape[0])
                if len(labels.shape)==3:
                    labels = labels.unsqueeze(1)
                count+=1
                intermediate_count += inputs.shape[0]

                inputs = inputs.to(device)
                labels = labels.to(device)
                #fix this line
                w = torch.nn.utils.parameters_to_vector([client.inc.parameters(), client.encoder_layers.parameters()])
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

                    torch.nn.utils.vector_to_parameters(w_r, server.parameters())
                    x1 = client(inputs, 'encode')
                    x2 = server(x1)
                    x3 = client(x2, 'decode')
                    
                    loss_right = loss_fxn.forward(x3, labels)

                    torch.nn.utils.vector_to_parameters(w_l, server.parameters())
                    x1 = client(inputs, 'encode')
                    x2 = server(x1)
                    x3 = client(x2, 'decode')

                    loss_left = loss_fxn.forward(x3, labels)

                    ghat = (loss_right - loss_left)/((2*ck)*perturb)
                    ghats.append(ghat)

                torch.cat(ghats, dim=0).mean(dim=0)
                if count==1:
                    m = ghat
                else:
                    m = momentum*m + ghat
                accum_ghat = ghat + momentum*m
                w = w - lr_server*accum_ghat
                torch.nn.utils.vector_to_parameters(w, server.parameters())

        if epoch%20==0:
            running_loss = 0.0
            running_dice = 0
            hds = []
            for inputs, labels,_,_ in dataloaders['train']:
                if len(labels.shape)==3:
                    labels = labels.unsqueeze(1)

                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.no_grad():
                    x1 = client(inputs, 'encode')
                    x2 = server(x1)
                    x3 = client(x2, 'decode')

                    loss=loss_fxn.forward(x3, labels)
                    # print("Reg loss: ",reg_loss)
                    
                    preds = (x3>=0.5)+0


                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    ri, ru = running_stats(labels,preds)
                    running_dice += dice_collated(ri,ru)
                    hd = compute_hd95(preds, labels)
                    if not math.isnan(hd):
                        hds.append(hd)
    
            epoch_loss = running_loss / ((len(dataset_dict['train'])))
            epoch_dice = running_dice / ((len(dataset_dict['train'])))
            print(f'Training server at epoch {epoch} Training Loss: {epoch_loss:.4f} Training Dice: {epoch_dice:.4f} HD95 avg: {torch.mean(torch.Tensor(hds))}')

            #save server
            if epoch_loss < best_tr_loss:
                best_tr_loss = epoch_loss
                torch.save(server.state_dict(), os.path.join(save_path,'server_best_tr.pth'))


    
def test(server, client, dataset_dict, device):
    server = server.to(device)
    client = client.to(device)
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
            x1 = client(img, 'encode')
            x2 = server(x1)
            x3 = client(x2, 'decode')
            preds = (x3>=0.5)

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
    epoch_dice = running_dice / dataset_dict['test']
    # epoch_dice = dice_coef(torch.cat(preds_all,axis=0),torch.cat(gold,axis=0))
    print(f'Testing {dataset_dict["name"]}: Dice: {epoch_dice:.4f} HD95 avg: {torch.mean(torch.Tensor(hds))}')

