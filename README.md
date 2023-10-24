# rDL-SIM-pytorch

## Points
- normal_flag：统一设置为0，直接除65535进行归一化，否则在预测过程中无法恢复图像
- exp_name：当前子项目使用的名称
- resume：引用的模型参数的子项目名称

## Environment
- CUDA 11.3
- Python 3.7
- Pytorch 1.10.0 : conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge