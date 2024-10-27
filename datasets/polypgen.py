import random
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data_transforms.kvasirSeg_transform import kvasirSeg_Transform


class PolypGen_Dataset(Dataset):
    def __init__(self, config, is_train=False, shuffle_list = True, apply_norm=True, is_test=False, center_num=1) -> None:
        super().__init__()
        self.root_path = config['root_path']
        self.img_names = []
        self.img_path_list = []
        self.label_path_list = []
        self.label_list = []
        self.is_train = is_train
        self.is_test = is_test
        self.center_num = center_num
        self.label_names = config['label_names']
        self.num_classes = len(self.label_names)
        self.config = config
        self.apply_norm = apply_norm
        self.palette = [[255,255,255]]

        if center_num=='super':
            self.populate_lists_super()
        else:
            self.populate_lists()
        if shuffle_list:
            p = [x for x in range(len(self.img_path_list))]
            random.shuffle(p)
            self.img_path_list = [self.img_path_list[pi] for pi in p]
            self.img_names = [self.img_names[pi] for pi in p]
            self.label_path_list = [self.label_path_list[pi] for pi in p]
            self.label_list = [self.label_list[pi] for pi in p]

        #define data transform
        self.data_transform = kvasirSeg_Transform(config=config)

    def __len__(self):
        return len(self.img_path_list)

    def populate_lists(self):
        if self.is_train:
            imgs_path = os.path.join(self.root_path, 'data_C'+str(self.center_num), 'train.txt')
        else:
            if self.is_test:
                imgs_path = os.path.join(self.root_path, 'data_C'+str(self.center_num), 'test.txt')
            else:
                imgs_path = os.path.join(self.root_path, 'data_C'+str(self.center_num), 'val.txt')

        imgs_root = os.path.join(self.root_path,'data_C'+str(self.center_num),'images_C'+str(self.center_num))
        masks_root = os.path.join(self.root_path,'data_C'+str(self.center_num),'masks_C'+str(self.center_num))

        im_list_file= open(imgs_path, 'r')
        im_list = im_list_file.readlines()

        for img in im_list:
            img = img.strip()
            # print(img)
            if (('jpg' not in img) and ('jpeg not in img') and ('png' not in img) and ('bmp' not in img)):
                continue
            
            self.img_names.append(img)
            self.img_path_list.append(os.path.join(imgs_root,img))
            self.label_path_list.append(os.path.join(masks_root, img[:-4]+"_mask.jpg"))
            self.label_list.append('Polyp')
    
    def populate_lists_super(self):
        if self.is_train:
            imgs_path = os.path.join(self.root_path, 'super_train.txt')
        else:
            if self.is_test:
                imgs_path = os.path.join(self.root_path, 'super_test.txt')
            else:
                imgs_path = os.path.join(self.root_path, 'super_val.txt')

        im_list_file= open(imgs_path, 'r')
        im_list = im_list_file.readlines()

        for img in im_list:
            img = img.strip()
            sl_idx = img.find('/')
            img_name = img[sl_idx+1:]
            data_num = img[:sl_idx]
            center_num = data_num[5:]

            if (('jpg' not in img) and ('jpeg not in img') and ('png' not in img) and ('bmp' not in img)):
                continue
            
            self.img_names.append(img_name)
            self.img_path_list.append(os.path.join(self.root_path,data_num,'images_'+center_num,img_name))
            self.label_path_list.append(os.path.join(self.root_path,data_num,'masks_'+center_num,img_name[:-4]+'_mask.jpg'))
            self.label_list.append('Polyp')


    def __getitem__(self, index):
        img = torch.as_tensor(np.array(Image.open(self.img_path_list[index]).convert("RGB")))
        if self.config['volume_channel']==2:
            img = img.permute(2,0,1)
            
        label = torch.Tensor(np.array(Image.open(self.label_path_list[index])))
        if len(label.shape)==3:
            label = label[:,:,0]
        
        label = label.unsqueeze(0)
        label = (label>0)+0
        label_of_interest = self.label_list[index]

        #convert all grayscale pixels due to resizing back to 0, 1
        img, label = self.data_transform(img, label, is_train=self.is_train, apply_norm=self.apply_norm)
        label = (label>=0.5)+0
        # label = label[0]

        return img.float(), label, self.img_names[index], label_of_interest
