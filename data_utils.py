from datasets.glas import GLAS_Dataset
from datasets.kvasirseg import KVASIRSEG_Dataset
from datasets.polypgen import PolypGen_Dataset
from datasets.isic import Skin_Dataset
from datasets.refuge import Refuge_Dataset
from datasets.rite import RITE_Dataset
from datasets.endovis import Endovis_Dataset
from datasets.chestxdet import ChestXDet_Dataset
from torch.utils.data import DataLoader
import yaml
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys
# sys.path.append('../endovis17')

def get_data(data_config, center_num=1):
    print(data_config)
    dataset_dict = {}
    dataset_sizes = {}
    dataloader_dict = {}

    if data_config['name']=='GLAS':
        dataset_dict['train'] = GLAS_Dataset(data_config, shuffle_list=True, is_train=True, apply_norm=data_config['use_norm'], no_text_mode=data_config['no_text_mode'])
        dataset_dict['val'] = GLAS_Dataset(data_config, shuffle_list=False, apply_norm=data_config['use_norm'], is_train=False, no_text_mode=data_config['no_text_mode'])
        dataset_dict['test'] = GLAS_Dataset(data_config, shuffle_list=False, is_train=False, is_test=True, apply_norm=data_config['use_norm'], no_text_mode=data_config['no_text_mode'])

        dataset_sizes['train'] = len(dataset_dict['train'])
        dataset_sizes['val'] = len(dataset_dict['val'])
        dataset_sizes['test'] = len(dataset_dict['test'])
        print(dataset_sizes)


        # dataloader_dict['train'] = DataLoader(dataset_dict['train'], data_config['batch_size'], shuffle=True)
        # dataloader_dict['val'] = DataLoader(dataset_dict['val'], data_config['batch_size'], shuffle=True)

    elif data_config['name']=='KVASIRSEG':
        dataset_dict['train'] = KVASIRSEG_Dataset(data_config, shuffle_list=True, is_train=True, apply_norm=data_config['use_norm'], no_text_mode=data_config['no_text_mode'])
        dataset_dict['val'] = KVASIRSEG_Dataset(data_config, shuffle_list=False, apply_norm=data_config['use_norm'], is_train=False, no_text_mode=data_config['no_text_mode'])
        dataset_dict['test'] = KVASIRSEG_Dataset(data_config, shuffle_list=False, is_train=False, is_test=True, apply_norm=data_config['use_norm'], no_text_mode=data_config['no_text_mode'])

        dataset_sizes['train'] = len(dataset_dict['train'])
        dataset_sizes['val'] = len(dataset_dict['val'])
        dataset_sizes['test'] = len(dataset_dict['test'])

        # dataloader_dict['train'] = DataLoader(dataset_dict['train'], data_config['batch_size'], shuffle=True)
        # dataloader_dict['val'] = DataLoader(dataset_dict['val'], data_config['batch_size'], shuffle=True)
        print(dataset_sizes)

    elif data_config['name']=='POLYPGEN':
        dataset_dict['train'] = PolypGen_Dataset(data_config, shuffle_list=True, is_train=True, apply_norm=data_config['use_norm'], center_num=center_num)
        dataset_dict['val'] = PolypGen_Dataset(data_config, shuffle_list=False, apply_norm=data_config['use_norm'], is_train=False, center_num=center_num)
        dataset_dict['test'] = PolypGen_Dataset(data_config, shuffle_list=False, is_train=False, is_test=True, apply_norm=data_config['use_norm'], center_num=center_num)
        dataset_dict['name'] = str(center_num)

        dataset_sizes['train'] = len(dataset_dict['train'])
        dataset_sizes['val'] = len(dataset_dict['val'])
        dataset_sizes['test'] = len(dataset_dict['test'])

        # dataloader_dict['train'] = DataLoader(dataset_dict['train'], data_config['batch_size'], shuffle=True)
        # dataloader_dict['val'] = DataLoader(dataset_dict['val'], data_config['batch_size'], shuffle=True)
        # print(dataset_sizes)


    elif data_config['name']=='SKIN':
        dataset_dict['train'] = Skin_Dataset(data_config, shuffle_list=True, is_train=True, apply_norm=data_config['use_norm'])
        dataset_dict['val'] = Skin_Dataset(data_config, shuffle_list=False, apply_norm=data_config['use_norm'], is_train=False)
        dataset_dict['test'] = Skin_Dataset(data_config, shuffle_list=False, is_train=False, is_test=True, apply_norm=data_config['use_norm'])

        dataset_sizes['train'] = len(dataset_dict['train'])
        dataset_sizes['val'] = len(dataset_dict['val'])
        dataset_sizes['test'] = len(dataset_dict['test'])

        # dataloader_dict['train'] = DataLoader(dataset_dict['train'], data_config['batch_size'], shuffle=True)
        # dataloader_dict['val'] = DataLoader(dataset_dict['val'], data_config['batch_size'], shuffle=True)
    
    elif data_config['name']=='REFUGE':
        dataset_dict['train'] = Refuge_Dataset(data_config, shuffle_list=True, is_train=True, apply_norm=data_config['use_norm'], no_text_mode=data_config['no_text_mode'])
        dataset_dict['val'] = Refuge_Dataset(data_config, shuffle_list=False, apply_norm=data_config['use_norm'], is_train=False, no_text_mode=data_config['no_text_mode'])
        dataset_dict['test'] = Refuge_Dataset(data_config, shuffle_list=False, is_train=False, is_test=True, apply_norm=data_config['use_norm'], no_text_mode=data_config['no_text_mode'])

        dataset_sizes['train'] = len(dataset_dict['train'])
        dataset_sizes['val'] = len(dataset_dict['val'])
        dataset_sizes['test'] = len(dataset_dict['test'])

        print(dataset_sizes)

        # dataloader_dict['train'] = DataLoader(dataset_dict['train'], data_config['batch_size'], shuffle=True)
        # dataloader_dict['val'] = DataLoader(dataset_dict['val'], data_config['batch_size'], shuffle=True)

    elif data_config['name']=='RITE':
        dataset_dict['train'] = RITE_Dataset(data_config, shuffle_list=True, is_train=True, apply_norm=data_config['use_norm'], no_text_mode=data_config['no_text_mode'])
        dataset_dict['val'] = RITE_Dataset(data_config, shuffle_list=False, apply_norm=data_config['use_norm'], is_train=False, no_text_mode=data_config['no_text_mode'])
        dataset_dict['test'] = RITE_Dataset(data_config, shuffle_list=False, is_train=False, is_test=True, apply_norm=data_config['use_norm'], no_text_mode=data_config['no_text_mode'])

        dataset_sizes['train'] = len(dataset_dict['train'])
        dataset_sizes['val'] = len(dataset_dict['val'])
        dataset_sizes['test'] = len(dataset_dict['test'])

        print(dataset_sizes)

        # dataloader_dict['train'] = DataLoader(dataset_dict['train'], data_config['batch_size'], shuffle=True)
        # dataloader_dict['val'] = DataLoader(dataset_dict['val'], data_config['batch_size'], shuffle=True)

    elif data_config['name']=='ENDOVIS':
        dataset_dict['train'] = Endovis_Dataset(data_config, shuffle_list=True, is_train=True, apply_norm=data_config['use_norm'], no_text_mode=data_config['no_text_mode'])
        dataset_dict['val'] = Endovis_Dataset(data_config, shuffle_list=False, apply_norm=data_config['use_norm'], is_train=False, no_text_mode=data_config['no_text_mode'])
        dataset_dict['test'] = Endovis_Dataset(data_config, shuffle_list=False, is_train=False, is_test=True, apply_norm=data_config['use_norm'], no_text_mode=data_config['no_text_mode'])

        dataset_sizes['train'] = len(dataset_dict['train'])
        dataset_sizes['val'] = len(dataset_dict['val'])
        dataset_sizes['test'] = len(dataset_dict['test'])

        print(dataset_sizes)

        # dataloader_dict['train'] = DataLoader(dataset_dict['train'], data_config['batch_size'], shuffle=True)
        # dataloader_dict['val'] = DataLoader(dataset_dict['val'], data_config['batch_size'], shuffle=True)

    elif data_config['name']=='CHESTXDET':
        dataset_dict['train'] = ChestXDet_Dataset(data_config, shuffle_list=True, is_train=True, apply_norm=data_config['use_norm'], no_text_mode=data_config['no_text_mode'])
        dataset_dict['val'] = ChestXDet_Dataset(data_config, shuffle_list=False, apply_norm=data_config['use_norm'], is_train=False, no_text_mode=data_config['no_text_mode'])
        dataset_dict['test'] = ChestXDet_Dataset(data_config, shuffle_list=False, is_train=False, is_test=True, apply_norm=data_config['use_norm'], no_text_mode=data_config['no_text_mode'])

        dataset_sizes['train'] = len(dataset_dict['train'])
        dataset_sizes['val'] = len(dataset_dict['val'])
        dataset_sizes['test'] = len(dataset_dict['test'])

        print(dataset_sizes)

    return dataset_dict

def visualize_PCA(dataset_dicts, phase='train'):
    images_centers = []
    c = ['red', 'green', 'blue', 'black', 'yellow', 'cyan']
    for i in range(len(dataset_dicts)):
        data = dataset_dicts[i][phase]
        images = []
        count = 0
        for img,_,_,_ in data:
            count+=1
            # if count>5:
            #     continue
            images.append(img)
        l = len(images)
        images = torch.stack(images,dim=0).permute(0,2,3,1).cpu().numpy().reshape((l,-1))

        # Scale data before applying PCA
        scaling=StandardScaler()
        
        # Use fit and transform method 
        scaling.fit(images)
        Scaled_data=scaling.transform(images)
        
        # Set the n_components=3
        principal=PCA(n_components=2)
        principal.fit(images)
        x=principal.transform(images)
        images_centers.append(x)

        #visualization
        plt.scatter(x=x[:,0], y=x[:,1],c=c[i])
    
    plt.legend(['C'+str(i+1) for i in range(len(dataset_dicts))])
    plt.savefig("./tmp2.png")



if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # dataset_dict = get_data(config, center_num=1)
    # print(dataset_dict['train'][0][0].shape)
    # print(dataset_dict['train'][0][1].shape)
    # print(dataset_dict['train'][52][1].any())
    # plt.imshow(dataset_dict['train'][0][0].permute(1,2,0))
    # plt.show()
    # plt.imshow(dataset_dict['train'][0][1], cmap='gray')
    # plt.show()
        
    dataset_dicts = [get_data(config, center_num=i) for i in [1, 2, 3, 4, 5, 6]]
    visualize_PCA(dataset_dicts)

