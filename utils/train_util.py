import cv2
import torch
import numpy as np
import math


def get_net_input_cuda(input_pol, input_coordinate, viewing_direction, ARGS):
    '''
        preprocessing the input data using gpu
        add positional encoding
        WARNING: I deleted phi_rgb for convenience, no sure the performance diff
    '''

    cat_list = []
    # import pdb; pdb.set_trace()
    if "onlyiun" in ARGS.netinput:
        cat_list.append(input_pol[:,4:5])
    if "fourraw" in ARGS.netinput:
        cat_list.append(input_pol[:,:4])
    if "fiveraw" in ARGS.netinput:
        cat_list.append(input_pol[:,:5])
    if "pol" in ARGS.netinput:
        phi = input_pol[:,5:6] # ablation on input as phi information
        
        # import time
        # torch.cuda.synchronize()
        # with torch.no_grad():
        # st = time.time()
        phi_encode = torch.cat([torch.cos(2 * phi), torch.sin(2 * phi)], axis=1)
        # torch.cuda.synchronize()
        # test_time = time.time()-st
        # print(test_time)
        # import pdb;pdb.set_trace()
        cat_list.append(input_pol[:,6:])
        if "rawphi" in ARGS.netinput:
            phi_encode = phi
        cat_list.append(phi_encode)
                        
    if "coord" in ARGS.netinput:
        cat_list.append(input_coordinate)
    if "vd" in ARGS.netinput: 
        cat_list.append(viewing_direction)

    net_input = torch.cat(cat_list, axis=1)   
    return net_input

def get_net_input_channel(ARGS):
    '''
        preprocessing the input data using gpu
        add positional encoding
        WARNING: I deleted phi_rgb for convenience, no sure the performance diff
    '''

    channel = 0
    # import pdb; pdb.set_trace()
    if "onlyiun" in ARGS.netinput:
        channel += 1
    if "fourraw" in ARGS.netinput:
        channel += 4
    if "fiveraw" in ARGS.netinput:
        channel += 5
    if "pol" in ARGS.netinput:
        if "rawphi" in ARGS.netinput:
            channel += 2
        else:
            channel += 3

    if "coord" in ARGS.netinput:
        channel +=3
    if "vd" in ARGS.netinput: 
        channel += 3
    if "embeddingpos" in ARGS.netinput: 
        channel += 3


    
    return channel


def adjust_learning_rate(optimizer, epoch, args):
    
    """Decay the learning rate based on schedule"""
    lr = args.lr
    args.warmup_epochs = 0
    if epoch < args.warmup_epochs:
        lr *=  float(epoch) / float(max(1.0, args.warmup_epochs))
        if epoch == 0 :
            lr = 1e-6
    else:
        # progress after warmup        
        if args.cos:  # cosine lr schedule
            # lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
            progress = float(epoch - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
            lr *= 0.5 * (1. + math.cos(math.pi * progress)) 
            # print("adjust learning rate now epoch %d, all epoch %d, progress"%(epoch, args.epochs))
        else:  # stepwise lr schedule
            for milestone in args.schedule:
                lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("Epoch-{}, base lr {}, optimizer.param_groups[0]['lr']".format(epoch, args.lr), optimizer.param_groups[0]['lr'])
