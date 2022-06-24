#%%
from time import time
import numpy as np
import random
import utils.dataloader_util as dataloader_util

import os, cv2
from torch.utils.data.dataset import Dataset
from glob import glob
from torch.utils.data import DataLoader

seed = 2020
np.random.seed(seed)
random.seed(seed)

# TODO reorganize the argument

class SfPDataset(Dataset):
    def __init__(self, 
                 dataroot, 
                 train = 1, 
                 split = "inter",
                 use_deepsfp = True,
                 use_sfpwild = True,
                 interpolated_normal = False,
                 fast = False,
                 with_viewing_dir = False,
                 crop_interval = 32
                 ) :
        super().__init__()
        self.dataroot = dataroot
        self.split = split
        self.use_deepsfp = use_deepsfp
        self.use_sfpwild = use_sfpwild
        self.interpolated_normal = interpolated_normal
        self.fast = fast
        self.with_viewing_dir = with_viewing_dir
        self.train = train
        self.crop_interval = crop_interval
        if self.split == "inter":
            train_pol_paths, train_normal_paths, train_mask_paths, \
            test_pol_paths, test_normal_paths, test_mask_paths = dataloader_util.inter_scenes_split(self)

        if train:
            self.pol_paths     =   train_pol_paths     
            self.normal_paths  =   train_normal_paths  
            self.mask_paths    =   train_mask_paths    
        else:
            self.pol_paths      =   test_pol_paths      
            self.normal_paths   =   test_normal_paths   
            self.mask_paths     =   test_mask_paths     

        print("training:", train)
        print("mask_paths", self.mask_paths[0])
        print("normal_paths", self.normal_paths[0])
        print("pol_paths", self.pol_paths[0])

        self.image_coordinate  = dataloader_util.get_coordinate(np.load(train_pol_paths[0]), viewing_dir=self.with_viewing_dir)
        self.viewing_direction = dataloader_util.get_vd()

    def __getitem__(self, index):
        '''
        cpu only do easy job like
        np.load input file, crop the file, random flip
        '''
        # import pdb; pdb.set_trace()
        net_in, vis_camera_normal, net_gt, net_mask, net_coordinate, viewing_direction \
            = dataloader_util.load_paired_data( \
            self.pol_paths[index], self.normal_paths[index], self.mask_paths[index], self.image_coordinate, self.viewing_direction, crop_interval=self.crop_interval)

        if self.train:
            net_in, vis_camera_normal, net_gt, net_mask, net_coordinate, viewing_direction  \
                    =  dataloader_util.crop_augmentation_list( \
                            [net_in, vis_camera_normal, net_gt, net_mask, net_coordinate, viewing_direction])

                
        sample_list = net_mask, net_in, vis_camera_normal, net_gt, net_coordinate, viewing_direction
        tensor_list = dataloader_util.ToTensor(sample_list)
        return tensor_list
    
    
    def __len__(self):
        
        return len(self.pol_paths)
    

    

class SfPTestDataset(Dataset):
    def __init__(self, 
                 dataroot,
                 txt_path=None,
                 with_viewing_dir=False,
                 use_deepsfp=None,
                 crop_interval=32
                 ) -> None:
        super().__init__()
        self.dataroot = dataroot
        self.with_viewing_dir = with_viewing_dir
        if txt_path:
            # txt_path = txt_path.replace("/home/chenyang/disk1/iccv2021-sfp-wild/data/iccv2021", dataroot)
            print(txt_path)
            with open(txt_path, 'r') as f:
                self.pol_paths = [line[:-1].replace("DATAROOT", dataroot) for line in f.readlines()]
        else:
            if use_deepsfp:
                self.pol_paths = sorted(glob("{}/eccv2020/test/*/*polar.npy".format(dataroot)))[1:]
                print("{}/eccv2020/test/*/*polar.npy".format(dataroot))
            else:
                self.pol_paths = sorted(glob("{}/*png".format(dataroot)))     

        self.image_coordinate = dataloader_util.get_coordinate(np.load(self.pol_paths[0]), viewing_dir=self.with_viewing_dir)
        self.viewing_direction = dataloader_util.get_vd()
        self.crop_interval=crop_interval

    def __getitem__(self, index):
        '''
        cpu only do easy job like
        np.load input file, crop the file, random flip
        '''

        net_in = np.load(self.pol_paths[index])
        h, w = net_in.shape[:2]
        h = h // self.crop_interval * self.crop_interval
        w = w // self.crop_interval * self.crop_interval
        net_in = net_in[:h,:w]
        net_coordinate = self.image_coordinate[0,:h,:w]
        viewing_direction = self.viewing_direction[0,:h,:w]
        
        sample_list =  net_in, net_coordinate, viewing_direction
        tensor_list = dataloader_util.ToTensor(sample_list)
        return tensor_list
    
    
    def __len__(self):
        
        return len(self.pol_paths)
import torch 

def create_dataloader(args):
    print('> Loading datasets ...')
    sfp_train_dataset   = SfPDataset(args.dataroot, 
                                    split=args.split,
                                    use_deepsfp = args.use_deepsfp,
                                    use_sfpwild = args.use_sfpwild,
                                    interpolated_normal=args.interpolated_normal,                                    
                                    train=1, fast=(args.training_mode=="fast"),
                                    crop_interval=args.crop_interval)
    sfp_test_dataset    = SfPDataset(args.dataroot, 
                                    split=args.split,
                                    use_deepsfp = args.use_deepsfp,
                                    use_sfpwild = args.use_sfpwild,
                                    interpolated_normal=args.interpolated_normal,                                    
                                    train=0, fast=(args.training_mode=="fast"),
                                    crop_interval=args.crop_interval)

    return sfp_train_dataset, sfp_test_dataset






if __name__=="__main__":
    sfp_train_dataset = SfPDataset("./data/iccv2021", train=1, with_flip_aug=True)
    return_list = sfp_train_dataset.__getitem__(0)
