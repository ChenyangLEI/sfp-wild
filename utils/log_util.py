import os
import time
import torch
import torchvision.utils as tutils
from tensorboardX import SummaryWriter
from collections import OrderedDict

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def init_logging(args):
    """ This logger copy all code, don't put too many things in code folder
        Initilizes the logging and the SummaryWriter modules
    Args:
        args ([type]): [description]
    Returns:
        writer: tensorboard SummaryWriter
        logger: open(log.txt)
    """
    
    
    os.makedirs(os.path.dirname(args.log_dir), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    logger = open(args.log_dir+'/log.log', 'a')
    code_path = os.path.join(args.log_dir,'code_copy')
    os.makedirs(code_path, exist_ok=True)
    print('copy all file in code foler: cp -r *py %s'%code_path)
    os.system('cp -r *.py %s'%code_path)
    os.system('cp -r *.sh %s'%code_path)
    os.system('cp -r utils %s'%code_path)
    os.system('cp -r models %s'%code_path)
    os.system('cp -r evaluation %s'%code_path)
    os.system('cp -r configs %s'%code_path)
    os.system('cp -r data_path_txt %s'%code_path)
    for k, v in vars(args).items():
        logger.write("\t{}: {}\n".format(k, v))
    logger.flush()
    return writer, logger

def save_model_checkpoint(model, epoch, save_path=None):
    """
    Saves a checkpoint under 'args['log_dir'] + '/ckpt.pth'
    """
    if hasattr(model, "module"):
        model = model.module
    
    save_dict = { \
        'state_dict': model.state_dict(), \
        'epoch':        epoch, \
        }
    
    # torch.save(model.state_dict(),  os.path.join(args.output_dir, 'net.pth'))
    # n=model.state_dict() is 300M, larger than tensorboard and log, I save to result folder
    torch.save(save_dict, save_path)
    del save_dict
    
def	resume_training(args, model, resumef=None):
    '''Resumes previous training or starts a new
    '''
    if resumef:
        print('resume training with ckpt %s'%resumef)
    else:
        print('use last epoch ckpt')
        # Saving code: torch.save(save_dict,           os.path.join(args.output_dir, 'ckpt.pth'))
        resumef =   os.path.join(args.output_dir, 'ckpt_best_val.pth')
        
    if os.path.isfile(resumef):
        if not args.multiprocessing_distributed:
#            checkpoint = torch.load(resumef)
#            state_dict = checkpoint['state_dict']
            loc = 'cuda'
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(resumef, map_location=loc)
        state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = "module." + k
            state_dict[name] = v



        print("> Resuming previous training")
        model.load_state_dict(state_dict)
        start_epoch = checkpoint['epoch']
        print("=> loaded checkpoint '{}' (epoch {})"\
                .format(resumef, start_epoch))
        
    else:
        print("No checkpoint at {}". format(resumef))
        start_epoch = 0

    return start_epoch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
