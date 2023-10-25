# rDL-SIM-pytorch

## Points
- normal_flag：统一设置为1，进行百分比归一化
- exp_name：当前子项目使用的名称
- resume：引用的模型参数的子项目名称
- model在并行dp的load中存在问题，多卡与单卡不通用，需要注意修改
- SR_dataloader数据经过归一化处理 DN_dataloader数据直接返回原始数据，因为要进行SI Pattern计算

## Environment
- CUDA 11.3
- Python 3.7
- Pytorch 1.10.0 : conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

## Questions
- 为什么DN_model的input_channels是nphases=3
- 论文中，SR模型在DN中不进行训练，但是代码中仍然存在SR的optimizer_sr
- TF代码中的ReduceLROnPlateau，需要在看看
- 先尝试在DN中不训练SR