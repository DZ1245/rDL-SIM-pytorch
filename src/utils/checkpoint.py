import os
import torch
import shutil
from collections import OrderedDict

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, exp_name, save_path, filename='checkpoint.pth'):
    """Saves checkpoint to disk"""
    directory = os.path.join(save_path, exp_name)
    # directory = "checkpoint/%s/" % (exp_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + '/model_best.pth')

def load_checkpoint(save_weights_path, resume_exp, exp_name, mode, model, optimizer=None, lr=None, local_rank=None, fix_loaded=False):
    if resume_exp is None:
        resume_exp = exp_name
    if mode == 'test' :
        load_name = os.path.join(save_weights_path, resume_exp, 'model_best.pth')
    else:
        load_name = os.path.join(save_weights_path, resume_exp, 'model_best.pth')
        # load_name = os.path.join(save_weights_path, resume_exp, 'checkpoint.pth')
    print("loading checkpoint %s" % load_name)
    
    # DDP 增加map_location参数
    if local_rank is None:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location='cuda:{}'.format(local_rank))

    start_epoch = checkpoint['epoch'] + 1
    min_loss = checkpoint['min_loss']

    if resume_exp != exp_name:
        start_epoch = 0
        min_loss = 1000.0

    # filter out different keys or those with size mismatch
    mismatch = False
    model_dict = model.state_dict()
    ckpt_dict = {}
    mismatch = False
    for k, v in checkpoint['state_dict'].items():
        if k in model_dict:
            if model_dict[k].size() == v.size():
                ckpt_dict[k.replace('.module', '')] = v
            else:
                print('Size mismatch while loading!   %s != %s   Skipping %s...'
                      % (str(model_dict[k].size()), str(v.size()), k))
                mismatch = True
        else:
            mismatch = True
    if len(model.state_dict().keys()) > len(ckpt_dict.keys()):
        mismatch = True
    
    # Overwrite parameters to model_dict
    model_dict.update(ckpt_dict)
    # Load to model
    model.load_state_dict(model_dict)

    # if size mismatch, give up on loading optimizer; if resuming from other experiment, also don't load optimizer
    if (not mismatch) and (optimizer is not None) and (resume_exp is not None):
        optimizer.load_state_dict(checkpoint['optimizer'])
        update_lr(optimizer, lr)

    # if fix_loaded:
    #     for k, param in model.named_parameters():
    #         if k in ckpt_dict.keys():
    #             print(k)
    #             param.requires_grad = False
    
    print("loaded checkpoint %s" % load_name)

    
    del checkpoint, ckpt_dict, model_dict
    
    return start_epoch, min_loss