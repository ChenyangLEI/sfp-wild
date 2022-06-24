import torch
import numpy as np
import torchvision

def get_loss(result, net_gt, loss_type):
    mask = torch.eq(torch.sum(net_gt, dim=1, keepdim=True), 0.0)
    mask = -mask.type(torch.float32) + 1.0
    if loss_type == 'mae':
        return mae_loss(result, net_gt, mask)
    elif loss_type == 'AL':
        return angular_loss(result,net_gt, mask)
    elif loss_type == 'TAL':
        return truncated_angular_loss(result,net_gt,mask)


def mae_loss(img1, img2, mask):
    mae_map = -torch.sum(img1 * img2, dim=1, keepdims=True) + 1
    loss_map = torch.abs(mae_map * mask)
    loss = torch.mean(loss_map)
    return loss

def angular_loss(img1,img2,mask):
    # img1 B*3*H*W 
    prediction_error = torch.cosine_similarity(img1, img2, dim=1, eps=1e-6)
    prediction_error = prediction_error[:,None,:,:]

    acos_mask = mask.float() \
            * (prediction_error.detach() < 0.999).float() * (prediction_error.detach() > -0.999).float()
    acos_mask = acos_mask > 0.0

 
    angular_loss = torch.mean(torch.acos(prediction_error[acos_mask]))

    return angular_loss

def truncated_angular_loss(img1, img2, mask):
    prediction_error = torch.cosine_similarity(img1, img2, dim=1, eps=1e-6)
    prediction_error = prediction_error[:,None,:,:]
    # Robust acos loss
    acos_mask = mask.float() \
            * (prediction_error.detach() < 0.9999).float() * (prediction_error.detach() > 0.0).float()
    cos_mask = mask.float() * (prediction_error.detach() <= 0.0).float()
    acos_mask = acos_mask > 0.0
    cos_mask = cos_mask > 0.0
    truncated_angular_loss = torch.sum(torch.acos(prediction_error*acos_mask)) - torch.sum(prediction_error*cos_mask)
    truncated_angular_loss = truncated_angular_loss / (torch.sum(cos_mask) + torch.sum(acos_mask))
    return truncated_angular_loss

def get_mae(net_gt, pred_camera_normal, net_mask):
 
    h, w = pred_camera_normal.shape[-2:]

    net_gt = net_gt[:, :, :h,:w]
    pred_camera_normal = pred_camera_normal[:, :, :h,:w]

    net_mask = net_mask[:, :, :h,:w]

    valid_mask = net_mask *  torch.where(torch.mean(net_gt, dim=1, keepdim=True) != 0 , torch.ones(torch.mean(net_gt, dim=1, keepdim=True).size()).cuda(), torch.zeros(torch.mean(net_gt, dim=1, keepdim=True).size()).cuda())
    mae_map = torch.sum(net_gt * pred_camera_normal, dim=1, keepdim=True)#.clip(-1,1)
    mae_map = torch.acos(mae_map) * 180. / np.pi

    b = net_gt.size(0)
    maes = []
    for i in range(b):
        mae_map_i = mae_map[i:i+1]
        valid_mask_i = valid_mask[i:i+1]
        mae = torch.mean(mae_map_i[(valid_mask_i * mae_map_i)>0])
        maes.append(mae.item())
    
    return np.mean(maes)
