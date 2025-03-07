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
from PIL import Image
# from cmaes import CMA


class Loss_fxn():
    def __init__(self, losses_list=[], ignore_idx=-1):
        if losses_list==[]:
            self.losses_list = [torch.nn.CrossEntropyLoss()]
        else:
            self.losses_list = losses_list
        self.ignore_idx = ignore_idx

    def forward(self, pred, label):
        tmp_wt = [1,1]
        loss = 0
        for i,l in enumerate(self.losses_list):
            try:
                loss += (tmp_wt[i]*l(pred, label, ignore_idx=self.ignore_idx))
            except:
                loss += (tmp_wt[i]*l(pred, label.float(), ignore_idx=self.ignore_idx))
        return loss

def train(server, client, dataset_dict, j, save_path, loss_string, device, bs=8, lr_server=1e-4, lr_client=0.005, ck=0.01, ignore_idx = -1, tmp_save_path = '.', num_epochs_client=51, num_epochs_server=26):
    server = server.to(device)
    client = client.to(device)
    os.makedirs(save_path, exist_ok=True)    
    #set up logger
    logging.basicConfig(filename=os.path.join(save_path,"training_progress.log"),
                    format='%(message)s',
                    filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    server_optimizer = optim.AdamW(server.parameters(), lr=float(lr_server))


    #set hyperparameters
    sp_avg = 5
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
    if 'CE' in loss_string:
        losses_list.append(nn.CrossEntropyLoss())
    loss_fxn = Loss_fxn(losses_list, ignore_idx=ignore_idx)
    print("debug: loss loaded")

    #train client
    server_optimizer.zero_grad()
    with torch.no_grad():
        server.eval()
        client.eval()
        for epoch in range(num_epochs_client):
            running_loss = 0.0
            running_intersection = 0
            running_union = 0
            running_corrects = 0
            running_dice = 0
            running_iou = 0
            intermediate_count = 0
            count = 0
            preds_all = []
            gold = []
            hds = []
            dataloaders = {}
            dataloaders['train'] = torch.utils.data.DataLoader(dataset_dict['train'], batch_size=bs, shuffle=True, num_workers=4)
            dataloaders['val'] = torch.utils.data.DataLoader(dataset_dict['val'], batch_size=bs, shuffle=False, num_workers=4)

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

            if epoch%5==4:
                running_loss = 0.0
                running_dice = 0
                intermediate_count = 0
                running_iou = 0
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
                        try:
                            x1 = client(inputs)
                        except ValueError as ve:
                            #if batchnorm messes up things
                            print("Avoiding batchnorm error")
                            x1 = client(inputs[:-1])
                            labels = labels[:-1]
                            inputs = inputs[:-1]

                        try:
                            outputs= server(x1)
                        except:
                            continue
                        loss=loss_fxn.forward(outputs, labels)
                        # print("Reg loss: ",reg_loss)
                        
                        # preds = (outputs>=0.5)
                        preds = outputs


                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        ri, ru = running_stats(labels,preds)
                        running_dice += dice_collated(ri,ru)
                        running_iou += no_blank_miou(labels, preds, ignore_idx=ignore_idx)
                        # hd = compute_hd95(preds, labels)
                        # if not math.isnan(hd):
                        #     hds.append(hd)
        
                #validation performance
                # print("HD95 avg: ", torch.mean(torch.Tensor(hds)))
                # print(running_loss, intermediate_count)
                # print(running_loss/intermediate_count)
                epoch_loss = running_loss / ((len(dataset_dict['val'])))
                epoch_dice = running_dice / ((len(dataset_dict['val'])))
                epoch_iou = running_iou / ((len(dataset_dict['val'])))
            
                # epoch_dice = dice_coef(torch.cat(preds_all,axis=0),torch.cat(gold,axis=0))
                print(f'Training client at epoch {epoch} Training Loss: {epoch_loss:.4f} Training mIoU: {epoch_iou:.4f}')

                #save client
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    torch.save(client.state_dict(), os.path.join(save_path,'client_best_val.pth'))
    
    #train server
    best_val_loss = 100000000
    best_tr_loss = 1000000000
    server.train()
    for epoch in range(num_epochs_server):
        running_loss = 0.0
        running_intersection = 0
        running_union = 0
        running_corrects = 0
        running_dice = 0
        running_iou = 0
        intermediate_count = 0
        count = 0
        preds_all = []
        gold = []
        hds = []
        dataloaders = {}
        dataloaders['train'] = torch.utils.data.DataLoader(dataset_dict['train'], batch_size=bs, shuffle=True, num_workers=4)
        dataloaders['val'] = torch.utils.data.DataLoader(dataset_dict['val'], batch_size=bs, shuffle=False, num_workers=4)
        server.train()
        for inputs, labels,text_idxs, text in dataloaders['train']:
            if len(labels.shape)==3:
                labels = labels.unsqueeze(1)
        # for inputs, labels,text_idxs, text, pt, pt_label in dataloaders[phase]:

            count+=1
            intermediate_count += inputs.shape[0]

            inputs = inputs.to(device)
            #avoid batchnorm error
            if inputs.shape[0]==1:
                continue
            labels = labels.to(device)

            # zero the parameter gradients
            server_optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                try:
                    x1 = client(inputs)
                except ValueError as ve:
                    #if batchnorm messes up things
                    print("Avoiding batchnorm error")
                    x1 = client(inputs[:-1])

            with torch.set_grad_enabled(True):
                try:
                    outputs = server(x1)
                    loss=loss_fxn.forward(outputs, labels)
                    # print("Reg loss: ",reg_loss)
                except ValueError as ve:
                    outputs = server(x1[:-1])
                    loss=loss_fxn.forward(outputs, labels[:-1])
                    labels = labels[:-1]
                    inputs = inputs[:-1]

                # backward + optimize only if in training phase
                loss.backward(retain_graph=False)
                server_optimizer.step()

            with torch.no_grad():
                # preds = (outputs>=0.5)
                preds = outputs
                # preds_all.append(preds.cpu())
                # gold.append(labels.cpu())
                # epoch_dice = dice_coef(preds,labels)
                # if count%100==0:
                #     print('iteration dice: ', epoch_dice)


                # statistics
                running_loss += loss.item() * inputs.size(0)
                ri, ru = running_stats(labels,preds)
                running_dice += dice_collated(ri,ru)
                running_iou += no_blank_miou(labels, preds, ignore_idx=ignore_idx)
                
        if epoch%5==0:
            print(count)
            print(running_loss, intermediate_count)
            print(running_loss/intermediate_count)
            epoch_loss = running_loss / (len(tr_dataset))
            epoch_dice = running_dice / ((len(tr_dataset)))
            epoch_iou = running_iou / ((len(tr_dataset)))
            # epoch_dice = dice_coef(torch.cat(preds_all,axis=0),torch.cat(gold,axis=0))
            print(f'Training at epoch {epoch} Train Loss: {epoch_loss:.4f} Train mIoU: {epoch_iou:.4f}') 

        if epoch%20==0:
            running_loss = 0.0
            running_dice = 0
            running_iou = 0
            intermediate_count = 0
            count = 0
            hds = []
            server.eval()
            for inputs, labels,text_idxs, text in dataloaders['val']:
            # for inputs, labels,text_idxs, text, pt, pt_label in dataloaders[phase]:
                if inputs.shape[0]==1:
                    break
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
                    
                    # preds = (outputs>=0.5)+0
                    preds = outputs

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    ri, ru = running_stats(labels,preds)
                    running_dice += dice_collated(ri,ru)
                    running_iou += no_blank_miou(labels, preds ,ignore_idx=ignore_idx)

                    # hd = compute_hd95(preds, labels)
                    # if not math.isnan(hd):
                    #     hds.append(hd)
    
            #validation performance
            # print("HD95 avg: ", torch.mean(torch.Tensor(hds)))
            # print(running_loss, intermediate_count)
            # print(running_loss/intermediate_count)
            epoch_loss = running_loss / ((len(dataset_dict['val'])))
            epoch_dice = running_dice / ((len(dataset_dict['val'])))
            epoch_iou = running_iou / ((len(dataset_dict['val'])))
        
            # epoch_dice = dice_coef(torch.cat(preds_all,axis=0),torch.cat(gold,axis=0))
            print(f'Training server at epoch {epoch} Validation Loss: {epoch_loss:.4f} Validation mIoU: {epoch_iou:.4f}') 
               
            #save server
            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                torch.save(server.state_dict(), os.path.join(save_path,'server_best_val.pth'))

    torch.save(server.state_dict(), tmp_save_path+'/tmp_server_bg.pth')
    torch.save(client.state_dict(), tmp_save_path+'/tmp_client_bg'+str(j)+'.pth')

    return server, client

def test(server, client, dataset_dict, device, ignore_idx=-1, vis_dir = './tmp_vis'):
    server = server.to(device).eval()
    client = client.to(device).eval()
    #make folder for saving 
    os.makedirs(vis_dir, exist_ok=True)
    #set up logger
    # logging.basicConfig(filename=os.path.join('saved_models',"testing_progress.log"),
    #                 format='%(message)s',
    #                 filemode='a')
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    bs = 1
    test_dataloader = torch.utils.data.DataLoader(dataset_dict['test'], batch_size=bs, shuffle=False, num_workers=4)
    hds = []
    count = 0
    running_dice = 0
    running_iou = 0
    count_idx = 0
    for img, label, img_name, txt in test_dataloader:
        count_idx+=1
        if count_idx%50!=0:
            continue
        print("img name: ", img_name)
        with torch.no_grad():
            if len(label.shape)==3:
                label = label.unsqueeze(1)
            img = img.to(device)
            label = label.to(device)
            count += bs
            x1 = client(img)
            outputs= server(x1)
            # preds = (outputs>=0.5)
            preds = outputs

            # statistics
            ri, ru = running_stats(label,preds)
            running_dice += dice_collated(ri,ru)
            running_iou += no_blank_miou(label, preds, ignore_idx=ignore_idx)
            # hd = compute_hd95(preds, label)
            # if not math.isnan(hd):
            #     hds.append(hd)
            #save figure
            try:
                save_visual(preds, label, dataset_dict['test'].palette, os.path.join(vis_dir,img_name))
            except:
                save_visual(preds, label, dataset_dict['test'].palette, os.path.join(vis_dir,img_name[0]))

    #validation performance
    # print("HD95 avg: ", torch.mean(torch.Tensor(hds)))
    # print(running_loss, intermediate_count)
    # print(running_loss/intermediate_count)
    epoch_dice = running_dice / len(dataset_dict['test'])
    epoch_iou = running_iou / len(dataset_dict['test'])
    # epoch_dice = dice_coef(torch.cat(preds_all,axis=0),torch.cat(gold,axis=0))
    print(f'Testing {dataset_dict["name"]}: mIoU: {epoch_iou:.4f}')

def save_visual(preds, label, palette, name):
    if len(label.shape)==4:
        label=label[0]
    if len(preds.shape)==4:
        preds = preds[0]
    if preds.shape[0]==1:
        vis_out = ((preds[0]>=0.5)+0)*255
    else:
        argmax_pred = torch.argmax(preds, dim=0)
        vis_out = torch.zeros(preds.shape[1], preds.shape[2], 3)
        assert preds.shape[0] == len(palette)
        for i in range(vis_out.shape[0]):
            for j in range(vis_out.shape[1]):
                if preds[argmax_pred[i][j]][i][j] <0.5:
                    vis_out[i][j] = torch.Tensor([0,0,0])
                else:
                    vis_out[i][j] = torch.Tensor(palette[argmax_pred[i,j]])
    vis_out = vis_out.cpu().numpy().astype(np.uint8)
    im = Image.fromarray(vis_out)
    if 'png' in name:
        im.save(name)
    else:
        im.save(name+'.png')