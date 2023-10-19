import os

import utils.config_SR as config_SR

##### Parse CmdLine Arguments #####
args, unparsed = config_SR.get_args()
cwd = os.getcwd()
print(args)

print(args.Dataset)