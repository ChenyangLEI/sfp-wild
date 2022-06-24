import numpy as np
from glob import glob
import os
import imageio
from tqdm import tqdm
import cv2
import argparse
import pandas as pd

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def anglar_within_angle(mae_map_valid, angle=11.25):
    return float(mae_map_valid[mae_map_valid < angle].shape[0]) / float(mae_map_valid.shape[0])
def mean_angular_error(mae_map, valid_mask):
    mae_map = np.abs(mae_map)
    print("mae_map.shape", mae_map.shape)
    mae_map_valid = mae_map[valid_mask>0]
    print("mae_map_valid.shape", mae_map_valid.shape)
    print("np.mean(mae_map_valid)", np.mean(mae_map_valid))
    print("np.median(median)", np.median(mae_map_valid))
    print("np.max(median)", np.max(mae_map_valid))
    print("np.sqrt(((mae_map_valid) ** 2).mean())", np.sqrt(((mae_map_valid) ** 2).mean()))
    print("anglar_within_angle(mae_map_valid, 11.25)", anglar_within_angle(mae_map_valid, 11.25) )
    print("anglar_within_angle(mae_map_valid, 22.5)", anglar_within_angle(mae_map_valid, 22.5) )
    print("anglar_within_angle(mae_map_valid, 30)", anglar_within_angle(mae_map_valid, 30) )

    return np.mean(mae_map_valid)

def get_metrics(gt_path, pred_path, mask_path):
    gt = np.load(gt_path)
    pred = np.load(pred_path) if "npy" in pred_path else imageio.imread(pred_path) / 255. * 2 - 1
    h, w = pred.shape[:2]
    gt = gt[:h,:w]
    pred = pred[:h,:w]
    pred = pred / (np.linalg.norm(pred, axis=2)[...,None] + 1e-10)

    if os.path.isfile(mask_path):
        mask = imageio.imread(mask_path) / 255.
        mask = mask[:h,:w]
    else:
        mask = np.ones_like(gt[...,0])
    valid_mask = mask *  np.where(np.mean(gt, axis=2) != 0 , 1, 0)
    mae_map = np.sum(gt * pred, axis=2).clip(-1,1)
    
    mae_map = np.arccos(mae_map) * 180. / np.pi
    mae_map_gray = np.uint8((mae_map*5*255.0*valid_mask/180).clip(0, 255))
    diff_color = cv2.applyColorMap(mae_map_gray, cv2.COLORMAP_JET)
    mae_map_valid = mae_map[(valid_mask * mae_map)>0]
    cropped_pred = np.uint8(valid_mask[..., None]*(pred+1)*0.5*255.0)
    cropped_pred_path = pred_path.replace(".png", "_cropped.png")
    cv2.imwrite(cropped_pred_path, cropped_pred[..., ::-1])
    print(cropped_pred_path)

 
    return (np.mean(mae_map_valid),
            diff_color,
            np.median(mae_map_valid), 
            np.sqrt(((mae_map_valid) ** 2).mean()), 
                anglar_within_angle(mae_map_valid, 11.25), 
                anglar_within_angle(mae_map_valid, 22.5), 
                anglar_within_angle(mae_map_valid, 30),
                mae_map*valid_mask, 
                valid_mask)
def get_coordinate(depth=None, viewing_dir=False):
    # h * w 
    h, w = 1024,1216
    u = (np.tile(np.arange(w),[h,1]) - w * 0.5)/ (0.5 * w)
    v = (np.tile(np.arange(h)[...,None],[1,w]) - 0.5 * h) / (0.5 * h)
    coordinate = np.concatenate([u[...,None],v[...,None], 1. * np.ones([h,w,1])],axis=2)
    print("u.min(), u.max()", u.min(), u.max())
    print("v.min(), v.max()", v.min(), v.max())
    return coordinate

parser = argparse.ArgumentParser(description="Train the normal estimation network")




# Dirs
parser.add_argument('--dataroot', default="../data/cvpr2022_camera_ready", type=str, help='path to dataset cvpr2022_camera_ready') 
parser.add_argument('--test_txt', default="../data_path_txt/sfpwild_test_inter_root.txt", type=str, help='path to dataset cvpr2022_camera_ready') 
parser.add_argument('--pred_root', default="../results/0727_resunet/sfpwild_test_intra", type=str, help='path to dataset cvpr2022_camera_ready') 

args = parser.parse_args()
if args.test_txt == "deepsfp.txt":
    gt_root = "{}/eccv2020".format(args.dataroot)
    pol_paths = sorted(glob("{}/test/*/*polar.npy".format(gt_root)))[1:]
else:
    print("test_txt", args.test_txt)
    gt_root = "{}/ready_kinect3_lucid_pair".format(args.dataroot)
    with open(args.test_txt, "r") as f:
        # FIXME
        pol_paths = [line[:-1].replace("DATAROOT", args.dataroot) for line in f.readlines()]
        print("first pol path:",pol_paths[0])

save_txt = args.pred_root + args.test_txt.split("/")[-1]
f = open(save_txt, "w")
print("pred_root", args.pred_root)
print("test_txt", args.test_txt)




p = get_coordinate()
error_sum = np.zeros_like(p[...,0])
mask_sum = np.zeros_like(p[...,0])

metrics = ["mean_ae", "median_ae", "rmse_ae", "ae_11", "ae_22", "ae_30"]
metrics_dict = {}
for metric in metrics:
    metrics_dict[metric] = []

# pol_path_list = []
# mean_ae_list = []
for idx, pol_path in enumerate(tqdm(pol_paths)):
    # ground truth is non-interpolated point cloud
    gt_path = pol_path.replace("_polar", "_normal") #should use this
#    gt_path = pol_path.replace("_polar", "_internorm")
    
    # ground truth is interpolated image
    # gt_path = pol_path.replace("polar", "internorm")

    pred_path = pol_path.replace(gt_root, args.pred_root).replace("npy","png")
    pred_diff_path = pred_path.replace(".png", "_diff.png")
    mask_path = pol_path.replace("polar.npy", "mask.png").replace("lucid_pair","lucid_pair_mask")
    
    (mean_ae, diff_color, median_ae, rmse_ae, ae_11, ae_22, ae_30, mae_map, valid_mask) = get_metrics(gt_path, pred_path, mask_path)
    print(pred_diff_path)
    imageio.imwrite(pred_diff_path, diff_color[...,::-1])
    error_sum += mae_map 
    mask_sum += valid_mask
    for idx, metric in enumerate([mean_ae, median_ae, rmse_ae, ae_11, ae_22, ae_30]):
        f.writelines("{} {}\n".format(pol_path, metric))
        metrics_dict[metrics[idx]].append(metric)
    
mean_error = error_sum / (mask_sum + 1e-10)
np.save(save_txt.replace("txt","npy"), mean_error)
print("\n\n")
for metric in metrics:
    print("{} {}: {}\n".format("average", metric, np.mean(metrics_dict[metric])))
    f.writelines("{} {}: {:.3f}\n".format("average", metric, np.mean(metrics_dict[metric])))
f.close()

df = pd.DataFrame({'path': pol_paths, 'mae': metrics_dict['mean_ae']})
csv_path = save_txt.replace(".txt", ".csv")
df.to_csv(csv_path)
