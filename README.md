# rDL-SIM-pytorch

## Points
- normal_flag：统一设置为1，进行百分比归一化
- exp_name：当前子项目使用的名称
- resume：引用的模型参数的子项目名称
- ~~model在并行dp的load中存在问题，多卡与单卡不通用，需要注意修改~~
- 修改并行为DDP模式，对loadcheckpoints需要增加参数map_location
- SR_dataloader数据经过归一化处理 DN_dataloader数据直接返回原始数据，因为要进行SI Pattern计算

## Environment
- CUDA 11.3
- Python 3.7
- Pytorch 1.10.0
- conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

## Questions
- 为什么DN_model的input_channels是nphases=3，因为代码将9通道图片分为3*3进行训练，不明白原因
- ~~论文中，SR模型在DN中不进行训练，但是代码中仍然存在SR的optimizer_sr，先在DN中不训练SR~~
- ~~TF代码中的ReduceLROnPlateau，需要在看看~~
- 直接采用pytorch中的学习率优化参数
- rDL-DN的predict中num_average的作用未知：获取光照模式??
- 原TF代码中，训练中cur_k0和modamp通过img_gt计算得到，然后在计算得到img_gen时使用了这些参数，存在问题cur_k0包含有gt信息?

## Datasets
- rDL_SIM_data/rDL_BioSR/rDL_SIM：train和val在细胞级划分，SR和DN均包括了所有数据
- rDL_SIM_data/rDL_BioSR/rDL_SIM_separate_v1：train和val在细胞级划分，SR和DN在切割好的图片中随机划分

## Log
- 2024.02.26:创建Test分支，开始尝试修改代码及过往疑问;处理pku数据，暂时只能划分9x9x55张128x128图片（论文说它划分了3w张），并根据论文划分数据集（SR三分之一，rDL三分之二）；**pku数据无法读取，rDL提供的读取代码不适用，尚未解决,暂时继续使用BioSR数据**
- 2024.02.27:重新按比例划分BioSR数据;**疑问：不同光照强度的相同图片被分开到train和val会影响吗**；