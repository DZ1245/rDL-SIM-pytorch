import argparse

arg_lists = []
parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Directory
dir_arg = add_argument_group('Directory')
dir_arg.add_argument('--root_path', type=str, default='/data/home/dz/rDL_SIM/SR/')
dir_arg.add_argument("--data_folder", type=str, default="Microtubules")
dir_arg.add_argument("--save_weights_path", type=str, default="./trained_models/SR_Inference_Module/")
dir_arg.add_argument("--save_weights_suffix", type=str, default="")

# Dataset
data_arg = add_argument_group('Dataset')
data_arg.add_argument('--dataset', type=str, default='Microtubules')

data_arg.add_argument('--input_height', type=int, default=128)
data_arg.add_argument('--input_width', type=int, default=128)
data_arg.add_argument('--input_channels', type=int, default=9)

data_arg.add_argument("--scale_factor", type=int, default=2)
data_arg.add_argument("--norm_flag", type=int, default=1)
data_arg.add_argument("--resize_flag", type=int, default=0)

# Model
model_arg = add_argument_group('Model')
model_arg.add_argument("--load_weights_flag", type=int, default=0)
model_arg.add_argument("--model_name", type=str, default="DFCAN")

# Training / test parameters
learn_arg = add_argument_group('Learning')

learn_arg.add_argument("--mixed_precision", type=str, default="1")
learn_arg.add_argument("--total_epoch", type=int, default=10000)
learn_arg.add_argument("--sample_epoch", type=int, default=1000)
learn_arg.add_argument("--validate_interval", type=int, default=2000)
learn_arg.add_argument("--validate_num", type=int, default=1000)
learn_arg.add_argument("--batch_size", type=int, default=4)
learn_arg.add_argument("--start_lr", type=float, default=1e-4)
learn_arg.add_argument("--lr_decay_factor", type=float, default=0.5)
learn_arg.add_argument('--beta1', type=float, default=0.9)
learn_arg.add_argument('--beta2', type=float, default=0.99)
learn_arg.add_argument('--ssim_weight', type=float, default=1e-2)

# Misc
misc_arg = add_argument_group('Misc')
# misc_arg.add_argument('--exp_name', type=str, default='exp')
# misc_arg.add_argument('--log_iter', type=int, default=20)
# misc_arg.add_argument('--log_dir', type=str, default='logs')
# misc_arg.add_argument('--data_dir', type=str, default='data')
misc_arg.add_argument('--num_gpu', type=int, default=1)
misc_arg.add_argument("--gpu_id", type=str, default="0")
misc_arg.add_argument('--random_seed', type=int, default=12345)
misc_arg.add_argument('--num_workers', type=int, default=5)
# misc_arg.add_argument('--use_tensorboard', action='store_true')
# misc_arg.add_argument('--viz', action='store_true', help='whether to save images')
# misc_arg.add_argument('--lpips', action='store_true', help='evaluates LPIPS if set true')

def get_args():
    """Parses all of the arguments above
    """
    args, unparsed = parser.parse_known_args()
    if args.num_gpu > 0:
        setattr(args, 'cuda', True)
    else:
        setattr(args, 'cuda', False)
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    return args, unparsed
