import random
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data_transforms.cityscapes_transform import CITYSCAPES_Transform


class CITYSCAPES_Dataset(Dataset):
    def __init__(self, config, is_train=False, shuffle_list = True, apply_norm=True, is_test=False, center_num=1, pure_test=False) -> None:
        super().__init__()
        self.root_path = config['root_path']
        self.img_names = []
        self.img_path_list = []
        self.label_path_list = []
        self.label_list = []
        self.is_train = is_train
        self.is_test = is_test
        self.pure_test = pure_test
        self.center_num = center_num
        self.label_names = config['label_names']
        self.num_classes = len(self.label_names)
        self.config = config
        self.apply_norm = apply_norm

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
        self.data_transform = CITYSCAPES_Transform(config=config)

    def __len__(self):
        return len(self.img_path_list)

    def populate_lists(self):
        if self.pure_test:
            imgs_path = os.path.join(self.root_path, 'all_test.txt')
            imgs_root = os.path.join(self.root_path,'leftImg8bit_trainvaltest','leftImg8bit','test')
            masks_root = os.path.join(self.root_path,'gtFine_trainvaltest','gtFine','test')
        elif self.is_train:
            imgs_path = os.path.join(self.root_path, 'centers_FL/center_'+str(self.center_num), 'train.txt')
            imgs_root = os.path.join(self.root_path,'leftImg8bit_trainvaltest','leftImg8bit','train')
            masks_root = os.path.join(self.root_path,'gtFine_trainvaltest','gtFine','train')

        else:
            if self.is_test:
                imgs_path = os.path.join(self.root_path, 'centers_FL/center_'+str(self.center_num), 'test.txt')
                imgs_root = os.path.join(self.root_path,'leftImg8bit_trainvaltest','leftImg8bit','train')
                masks_root = os.path.join(self.root_path,'gtFine_trainvaltest','gtFine','train')
            else:
                imgs_path = os.path.join(self.root_path, 'centers_FL/center_'+str(self.center_num), 'val.txt')
                imgs_root = os.path.join(self.root_path,'leftImg8bit_trainvaltest','leftImg8bit','train')
                masks_root = os.path.join(self.root_path,'gtFine_trainvaltest','gtFine','train')

        im_list_file= open(imgs_path, 'r')
        im_list = im_list_file.readlines()

        for img in im_list:
            img = img.strip()
            # print(img)
            if (('jpg' not in img) and ('jpeg not in img') and ('png' not in img) and ('bmp' not in img)):
                continue
            
            self.img_names.append(img)
            self.img_path_list.append(os.path.join(imgs_root,img[:img.find('_')],img))
            self.label_path_list.append(os.path.join(masks_root, img[:img.find('_')], img[:-16]+"_gtFine_color.png"))
            self.label_list.append('')
    
    def populate_lists_super(self):
        if self.pure_test:
            imgs_paths = [os.path.join(self.root_path, 'all_test.txt')]
            imgs_root = os.path.join(self.root_path,'leftImg8bit_trainvaltest','leftImg8bit','test')
            masks_root = os.path.join(self.root_path,'gtFine_trainvaltest','gtFine','test')
        elif self.is_train:
            imgs_paths = [os.path.join(self.root_path, 'centers_FL/center_'+str(center_num), 'train.txt') for center_num in range(1,19)]
            imgs_root = os.path.join(self.root_path,'leftImg8bit_trainvaltest','leftImg8bit','train')
            masks_root = os.path.join(self.root_path,'gtFine_trainvaltest','gtFine','train')

        else:
            if self.is_test:
                imgs_paths = [os.path.join(self.root_path, 'centers_FL/center_'+str(center_num), 'test.txt') for center_num in range(1,19)]
                imgs_root = os.path.join(self.root_path,'leftImg8bit_trainvaltest','leftImg8bit','train')
                masks_root = os.path.join(self.root_path,'gtFine_trainvaltest','gtFine','train')
            else:
                imgs_paths = [os.path.join(self.root_path, 'centers_FL/center_'+str(center_num), 'val.txt') for center_num in range(1,19)]
                imgs_root = os.path.join(self.root_path,'leftImg8bit_trainvaltest','leftImg8bit','train')
                masks_root = os.path.join(self.root_path,'gtFine_trainvaltest','gtFine','train')

        for imgs_path in imgs_paths:
            im_list_file= open(imgs_path, 'r')
            im_list = im_list_file.readlines()

            for img in im_list:
                img = img.strip()
                # print(img)
                if (('jpg' not in img) and ('jpeg not in img') and ('png' not in img) and ('bmp' not in img)):
                    continue
                
                self.img_names.append(img)
                self.img_path_list.append(os.path.join(imgs_root,img[:img.find('_')],img))
                self.label_path_list.append(os.path.join(masks_root, img[:img.find('_')], img[:-16]+"_gtFine_color.png"))
                self.label_list.append('')


    def __getitem__(self, index):
        img = torch.as_tensor(np.array(Image.open(self.img_path_list[index]).convert("RGB")))
        if self.config['volume_channel']==2:
            img = img.permute(2,0,1)
            
        label = torch.Tensor(np.array(Image.open(self.label_path_list[index])))[:,:,:-1]
        
        #convert label into one hot encodings
        palette = self.config['palette']
        num_labels = len(palette)
        label_ohe = torch.zeros(num_labels, label.shape[0], label.shape[1])

        for i in range(num_labels):
            label_ohe[i] = torch.all(torch.eq(label, torch.Tensor(palette[i])),dim=2)
        
        # label_ohe = label_ohe.unsqueeze(0)
        label_ohe = (label_ohe)+0
        label_of_interest = ''

        #convert all grayscale pixels due to resizing back to 0, 1
        img, label_ohe = self.data_transform(img, label_ohe, is_train=self.is_train, apply_norm=self.apply_norm)
        label_ohe = (label_ohe>=0.5)+0
        # label_ohe = label_ohe[0]

        return img, label_ohe, self.img_names[index], label_of_interest
