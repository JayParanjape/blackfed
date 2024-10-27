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
from data_utils import get_data
import yaml
from PIL import Image


#define the loss function based on the input loss string
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


#training baselines includes the model without any split learning
def train(model, dataset_dict, save_path, loss_string, device, num_epochs = 2000, center_num=0, bs=8):
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
    if 'CE' in loss_string:
        losses_list.append(nn.CrossEntropyLoss())
    loss_fxn = Loss_fxn(losses_list)
    # print("debug: loss loaded, device: ", device)

    #control parameter for doing only testing

    #set hyperparameters
    lr = 1e-4

    tr_dataset, val_dataset, test_dataset = dataset_dict['train'], dataset_dict['val'], dataset_dict['test']
    best_val_loss = 10000
    best_tr_loss = 10000
    # print("debug: len tr dataset", len(tr_dataset))

    #train model
    best_val_loss = 100000000
    best_tr_loss = 1000000000
    for epoch in range(1,num_epochs+1):
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
                try:
                    outputs = model(inputs)
                except ValueError as ve:
                    outputs = model(inputs[:-1])
                    inputs = inputs[:-1]
                    labels = labels[:-1]
                loss=loss_fxn.forward(outputs, labels)
                # print("Reg loss: ",reg_loss)
                
                # backward + optimize only if in training phase
                loss.backward(retain_graph=False)
                optimizer.step()

            with torch.no_grad():
                # preds = (outputs>=0.5)
                preds = outputs

                # statistics
                running_loss += loss.item() * inputs.size(0)
                ri, ru = running_stats(labels,preds)
                running_dice += dice_collated(ri,ru)
                running_iou += no_blank_miou(labels, preds)

        #check training performance
        if epoch%5==0:
            epoch_loss = running_loss / ((len(tr_dataset)))
            epoch_dice = running_dice / ((len(tr_dataset)))
            epoch_iou = running_iou / ((len(tr_dataset)))            
            print(f'Training at epoch {epoch} Train Loss: {epoch_loss:.4f} Train Dice: {epoch_dice:.4f}') 

        #check validation performance
        if epoch%5==0:
            running_loss = 0.0
            running_dice = 0
            running_iou = 0
            intermediate_count = 0
            count = 0
            hds = []
            for inputs, labels,text_idxs, text in dataloaders['val']:
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
                    
                    # preds = (outputs>=0.5)
                    preds = outputs

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    ri, ru = running_stats(labels,preds)
                    running_dice += dice_collated(ri,ru)
                    running_iou += no_blank_miou(labels, preds)
                    # hd = compute_hd95(preds, labels)
                    # if not math.isnan(hd):
                    #     hds.append(hd)
    
            #validation performance
            epoch_loss = running_loss / ((len(dataset_dict['val'])))
            epoch_dice = running_dice / ((len(dataset_dict['val'])))
            epoch_iou = running_iou / ((len(dataset_dict['val'])))
            
            print(f'Validating model at epoch {epoch} Validation Loss: {epoch_loss:.4f} Validation Dice: {epoch_dice:.4f}') 
            
            #save model
            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                torch.save(model.state_dict(), os.path.join(save_path,'client_'+str(int(center_num)-1)+'_best_val.pth'))
    return model
    
def save_visual(preds, label, palette, name):
    if len(label.shape)==4:
        label=label[0]
    if len(preds.shape)==4:
        preds = preds[0]
    argmax_pred = torch.argmax(preds, dim=0)
    vis_out = torch.zeros(preds.shape[1], preds.shape[2], 3)
    assert preds.shape[0] == len(palette)
    for i in range(vis_out.shape[0]):
        for j in range(vis_out.shape[1]):
            vis_out[i][j] = torch.Tensor(palette[argmax_pred[i,j]])
    vis_out = vis_out.cpu().numpy().astype(np.uint8)
    im = Image.fromarray(vis_out)
    if 'png' in name:
        im.save(name)
    else:
        im.save(name+'.png') 
    
def test(model, dataset_dict, load_path, loss_string, device, center_num='0', vis_dir='tmp_vis'):
    #test on test dataset
    model.load_state_dict(torch.load(load_path), strict=True)
    model = model.to(device).eval()
    os.makedirs(vis_dir, exist_ok=True)

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
    loss_fxn = Loss_fxn(losses_list)
    # print("debug: loss loaded")

    running_loss = 0.0
    running_dice = 0
    intermediate_count = 0
    running_iou = 0.0
    count = 0
    hds = []
    bs=1
    dataloader = torch.utils.data.DataLoader(dataset_dict['test'], batch_size=bs, shuffle=True, num_workers=4)

    for inputs, labels,img_name, text in dataloader:
    # for inputs, labels,text_idxs, text, pt, pt_label in dataloaders[phase]:
        count+=1
        intermediate_count += inputs.shape[0]

        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.no_grad():
            outputs = model(inputs)
            # print("Outputs.shape: ", outputs.shape)
            loss=loss_fxn.forward(outputs, labels)
            # print("Reg loss: ",reg_loss)
            
            # preds = (outputs>=0.5)
            preds = outputs
            try:
                save_visual(preds, labels, dataset_dict['test'].palette, os.path.join(vis_dir,img_name))
            except:
                save_visual(preds, labels, dataset_dict['test'].palette, os.path.join(vis_dir,img_name[0]))

            # statistics
            running_loss += loss.item() * inputs.size(0)
            ri, ru = running_stats(labels,preds)
            running_dice += dice_collated(ri,ru)
            running_iou += no_blank_miou(labels, preds)
            # hd = compute_hd95(preds, labels)
            # if not math.isnan(hd):
            #     hds.append(hd)

    epoch_loss = running_loss / ((len(dataset_dict['test'])))
    epoch_dice = running_dice / ((len(dataset_dict['test'])))
    epoch_iou = running_iou / ((len(dataset_dict['test'])))
    # epoch_dice = dice_coef(torch.cat(preds_all,axis=0),torch.cat(gold,axis=0))
    print(f'Testing model on center {str(center_num)} Test Loss: {epoch_loss:.4f} Test mIoU: {epoch_iou:.4f} ') 

    return model


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
    
    print(device, center_num, only_test, fed_learning, config_file, num_epochs, load_path)

    model = SegFormer(n_channels=64, n_classes=32, config_path='/mnt/store/jparanj1/blackfed/mmsegmentation/configs/segformer/segformer_blackfed_camvid.py')
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
        model = train(model, dataset_dict, save_path = './saved_models' , loss_string='dice', device=device, num_epochs=num_epochs, center_num=center_num, bs=4)
        model = test(model, dataset_dict, load_path = './saved_models/client_'+str(int(center_num)-1)+'_best_val.pth', loss_string='dice', device=device)
    else:
        #only test - give load path for model instead of save path
        # model = test(model, dataset_dict, load_path = './skin_baseline_'+str(center_num)+'/model_best_val.pth', loss_string='bce + dice', device=device)
        # model = test(model, dataset_dict, load_path = './baselines/polyp/polyp_baseline_'+str(2)+'/model_best_val.pth', loss_string='bce + dice', device=device)
        # model = test(model, dataset_dict, load_path = './saved_models/client_'+str(int(center_num)-1)+'_best_val.pth', loss_string='bce + dice', device=device)
        cl_idx = load_path.find('client')
        model = test(model, dataset_dict, load_path = load_path, loss_string='dice', device=device, center_num=center_num, vis_dir=load_path[cl_idx:9+cl_idx]+"_dataset_"+str(center_num))

