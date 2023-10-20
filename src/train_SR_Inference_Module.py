import os

import utils.config_SR as config_SR

##### Parse CmdLine Arguments #####
args, unparsed = config_SR.get_args()
cwd = os.getcwd()

root_path = args.root_path
data_folder = args.data_folder
save_weights_path = args.save_weights_path
save_weights_suffix = args.save_weights_suffix

