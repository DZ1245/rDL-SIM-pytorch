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

def load_checkpoint(save_weights_path, resume_exp, exp_name, mode, model, optimizer, lr, model_type='SR', fix_loaded=False):
    if resume_exp is None:
        resume_exp = exp_name
    if mode == 'test' :
        load_name = os.path.join(save_weights_path, resume_exp, 'model_best.pth')
    else:
        #load_name = os.path.join('checkpoint', args.resume_exp, 'model_best.pth')
        load_name = os.path.join(save_weights_path, resume_exp, 'checkpoint.pth')
    print("loading checkpoint %s" % load_name)
    checkpoint = torch.load(load_name)
    start_epoch = checkpoint['epoch'] + 1
    if resume_exp != exp_name:
        start_epoch = 0

    # filter out different keys or those with size mismatch
    model_dict = OrderedDict()
    for k,v in checkpoint['state_dict'].items():
        model_dict[k.replace('module.','')] = v
    # model_dict = model.state_dict()
    # ckpt_dict = {}
    mismatch = False
    # for k, v in checkpoint['state_dict'].items():
    #     if k in model_dict:
    #         if model_dict[k].size() == v.size():
    #             ckpt_dict[k] = v
    #         else:
    #             print('Size mismatch while loading!   %s != %s   Skipping %s...'
    #                   % (str(model_dict[k].size()), str(v.size()), k))
    #             mismatch = True
    #     else:
    #         mismatch = True
    # if len(model.state_dict().keys()) > len(ckpt_dict.keys()):
    #     mismatch = True
    # # Overwrite parameters to model_dict
    # model_dict.update(ckpt_dict)
    # # Load to model
    # print(mismatch)
    model.load_state_dict(model_dict, strict=True)
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
    # del checkpoint, ckpt_dict, model_dict
    del checkpoint, model_dict
    return start_epoch