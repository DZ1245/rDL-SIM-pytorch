import os
import torch
import shutil

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