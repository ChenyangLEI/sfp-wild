import numpy as np
import os
import cv2
from glob import glob
import torch
from tqdm import tqdm

def get_coordinate(depth, viewing_dir=False):
    # h * w 
    h, w = depth.shape[:2]
    u = (np.tile(np.arange(w),[h,1]) - w * 0.5)/ (0.5 * w)
    v = (np.tile(np.arange(h)[...,None],[1,w]) - 0.5 * h) / (0.5 * h)
    coordinate = np.concatenate([u[...,None],v[...,None], 1. * np.ones([h,w,1])],axis=2)
    if viewing_dir:
        coordinate = np.load("./utils/vd_local.npy")
    return coordinate[None,...]

def get_vd(path="./utils/vd_local.npy"):
    return np.load(path)[None, ...]
    
    # interpolated_normal, use_sfpwild, use_deepsfp



# interpolated_normal, use_sfpwild, use_deepsfp
# debug fast


def get_test_path_from_txt_old(dataroot = "../data/iccv2021",
            test_txt = "../data/iccv2021/sfpwild_test_inter.txt",
            internorm = False,
            ):
    with open(test_txt, "r") as f: 
        pol_paths = [line[:-1].replace("/home/chenyang/disk1/iccv2021-sfp-wild/data/iccv2021", dataroot)
                        for line in f.readlines()]
    gt_path_list = []
    mask_path_list = []

    for idx, pol_path in enumerate(pol_paths):
        gt_path = pol_path.replace("polar", "normal")
        if internorm:
            gt_path = pol_path.replace("polar", "internorm")
        mask_path = pol_path.replace("polar.npy", "mask.png").replace("lucid_pair","lucid_pair_mask")
        gt_path_list.append(gt_path)
        mask_path_list.append(mask_path)
    return pol_paths, gt_path_list, mask_path_list

def get_test_path_from_txt(dataroot = "../data/iccv2021",
            test_txt = "../data/iccv2021/sfpwild_test_inter.txt",
            internorm = False,
            ):
    with open(test_txt, "r") as f: 
        pol_paths = [line[:-1].replace("DATAROOT", dataroot)
                        for line in f.readlines()]
    gt_path_list = []
    mask_path_list = []

    for idx, pol_path in enumerate(pol_paths):
        gt_path = pol_path.replace("polar", "normal")
        if internorm:
            gt_path = pol_path.replace("polar", "internorm")
        mask_path = pol_path.replace("polar.npy", "mask.png").replace("lucid_pair","lucid_pair_mask")
        gt_path_list.append(gt_path)
        mask_path_list.append(mask_path)
    return pol_paths, gt_path_list, mask_path_list

def write(path_list, path_txt = "../data/iccv2021/sfpwild_test_inter.txt",
            ):
    with open(path_txt, "w+") as f: 
        for path_item in path_list:
            f.write(path_item+'\n')
            # pol_paths = [line[:-1].replace("/home/chenyang/disk1/iccv2021-sfp-wild/data/iccv2021", dataroot)
            #                 for line in f.readlines()]
    return 

def inter_scenes_split(ARGS):
    test_pol_paths, test_normal_paths, test_mask_paths = ([], [], [])
    train_pol_paths, train_normal_paths, train_mask_paths = ([], [], [])
    if ARGS.use_sfpwild:
        test_pol_paths, test_normal_paths, test_mask_paths = get_test_path_from_txt(
                dataroot = ARGS.dataroot,
                test_txt = "./data_path_txt/sfpwild_test_inter_root.txt",
                internorm=ARGS.interpolated_normal
        )
        train_pol_paths, train_normal_paths, train_mask_paths = get_test_path_from_txt(
                dataroot = ARGS.dataroot,
                test_txt = "./data_path_txt/sfpwild_train_inter_root.txt",
                internorm=ARGS.interpolated_normal
                )  
    print("test_pol_paths in sfpwild", len(test_pol_paths))
    print("train_pol_paths in sfpwild", len(train_pol_paths))

    if ARGS.use_deepsfp:
        train_pol_paths += sorted(glob(ARGS.dataroot+"/eccv2020/train/*/*polar.npy"))
        train_normal_paths += sorted(glob(ARGS.dataroot+"/eccv2020/train/*/*normal.npy"))
        train_mask_paths += sorted(glob(ARGS.dataroot+"/eccv2020/train/*/*polar.npy"))
        test_pol_paths += sorted(glob(ARGS.dataroot+"/eccv2020/test/*/*polar.npy"))[1:]
        test_normal_paths += sorted(glob(ARGS.dataroot+"/eccv2020/test/*/*normal.npy"))[1:]
        test_mask_paths += sorted(glob(ARGS.dataroot+"/eccv2020/test/*/*polar.npy"))[1:]
        print("Deepsfp", len(train_pol_paths), len(test_pol_paths))
    

    interval = 1
    if ARGS.fast:
        interval = 20
    train_pol_paths = train_pol_paths[::interval]
    train_normal_paths = train_normal_paths[::interval]
    train_mask_paths =train_mask_paths[::interval]
    test_pol_paths = test_pol_paths[::interval]
    test_normal_paths = test_normal_paths[::interval]
    test_mask_paths =test_mask_paths[::interval]    
    return  train_pol_paths, train_normal_paths, train_mask_paths, test_pol_paths, test_normal_paths, test_mask_paths


def load_paired_data(pol_path, normal_gt_path, mask_path, image_coordinate, viewing_direction, crop_interval=32):
    net_in = np.load(pol_path)
    h, w = net_in.shape[:2]
    h = h // crop_interval * crop_interval
    w = w // crop_interval * crop_interval

    normal_inter_path = normal_gt_path.replace('normal.npy','internorm.npy')
    net_gt = np.load(normal_gt_path)
    if "eccv" in normal_inter_path:
        vis_camera_normal = net_gt
        net_mask = np.ones([h,w,1])
    else:
        vis_camera_normal = np.load(normal_inter_path)
        if cv2.imread(mask_path) is None:
            net_mask = np.ones_like(net_gt[..., 0:1])
        else:
            net_mask = cv2.imread(mask_path) / 255.

    net_in = net_in[:h,:w]
    net_gt = net_gt[:h,:w]
    vis_camera_normal = vis_camera_normal[:h,:w]
    net_coordinate = image_coordinate[0,:h,:w]
    viewing_direction = viewing_direction[0,:h,:w]
    net_mask = net_mask[:h,:w,0:1]
    net_gt = net_gt * net_mask
    vis_camera_normal = vis_camera_normal * net_mask

    return net_in, vis_camera_normal, net_gt, net_mask, net_coordinate, viewing_direction


    

def crop_augmentation_list(im_list, center_crop=False, patch_size=512):
    # Crop
    magic = np.random.random()
    if magic<0.5:
        h_orig,w_orig = im_list[0].shape[0:2]   # h, w, c
        h_crop = patch_size*2
        w_crop = patch_size*2
        try:
            w_offset = 0    #np.random.randint(0, w_orig-w_crop-1)
            h_offset = 0    #np.random.randint(0, h_orig-h_crop-1)
        except:
            print("Original W %d, desired W %d"%(w_orig,w_crop))
            print("Original H %d, desired H %d"%(h_orig,h_crop))
        return [img[h_offset:h_offset+h_crop:2,w_offset:w_offset+w_crop:2,:] for img in im_list] #im_in, im_GT
    else:
        h_orig,w_orig = im_list[0].shape[0:2]
        h_crop = patch_size
        w_crop = patch_size
        try:
            w_offset = np.random.randint(0, w_orig-w_crop-1)
            h_offset = np.random.randint(0, h_orig-h_crop-1)
        except:
            print("Original W %d, desired W %d"%(w_orig,w_crop))
            print("Original H %d, desired H %d"%(h_orig,h_crop))
        if center_crop:
            h_offset = h_orig // 2 - 256
            w_offset = w_orig // 2 - 256
        return [img[h_offset:h_offset+h_crop,w_offset:w_offset+w_crop,:] for img in im_list] #im_in, im_GT

def crop_augmentation_list_deepsfp(im_list, ):
    
    h_orig, w_orig = im_list[0].shape[1:3]   # n, h, w, c
    h_crop = 256
    w_crop = 256
    try:
        w_offset = np.random.randint(w_crop, w_orig-2*w_crop-1) # we use w_orig-2*w_crop-1 to try to crop center area
        h_offset = np.random.randint(h_crop, h_orig-2*h_crop-1) # or there will be too much invalid pixels
        # print('use crop size 256')
    except:
        print("Original W %d, desired W %d"%(w_orig,w_crop))
        print("Original H %d, desired H %d"%(h_orig,h_crop))
    return [img[:, h_offset:h_offset+h_crop, w_offset:w_offset+w_crop, :] for img in im_list]



def ToTensor(sample_list):
    """Convert ndarrays in sample to Tensors."""

    def SampleToTensor(sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        # image = image.transpose((2, 0, 1))
        sample_tensor = torch.from_numpy(sample.copy().astype(np.float32)).permute(2,0,1)
        
        return sample_tensor
    
    return [SampleToTensor(sample) for sample in sample_list]
