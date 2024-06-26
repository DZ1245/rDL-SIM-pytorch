# rDL-SIM-pytorch

## Environment
- CUDA 11.3
- Python 3.7
- Pytorch 1.10.0
- conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

## Points
- normal_flag：统一设置为1，进行百分比归一化
- exp_name：当前子项目使用的名称
- resume：引用的模型参数的子项目名称
- ~~model在并行dp的load中存在问题，多卡与单卡不通用，需要注意修改~~
- 修改并行为DDP模式，对loadcheckpoints需要增加参数map_location
- SR_dataloader数据经过归一化处理 DN_dataloader数据直接返回原始数据，因为要进行SI Pattern计算

## Questions
- 为什么DN_model的input_channels是nphases=3，因为代码将9通道图片分为3*3进行训练，不明白原因
- ~~论文中，SR模型在DN中不进行训练，但是代码中仍然存在SR的optimizer_sr，先在DN中不训练SR~~
- ~~TF代码中的ReduceLROnPlateau，需要在看看~~
- ~~直接采用pytorch中的学习率优化参数~~
- rDL-DN的predict中num_average的作用未知：获取光照模式??
- 原TF代码中，训练中cur_k0和modamp通过img_gt计算得到，然后在计算得到img_gen时使用了这些参数，存在问题cur_k0包含有gt信息?
- FFT中的振铃校正（apodization）

## Datasets
- rDL_SIM_data/rDL_BioSR/rDL_SIM：train和val在细胞级划分，SR和DN均包括了所有数据
- rDL_SIM_data/rDL_BioSR/rDL_SIM_separate_v1：train和val在细胞级划分，SR和DN在切割好的图片中随机划分

## Log
- 2024.02.26:创建Test分支，开始尝试修改代码及过往疑问;处理pku数据，暂时只能划分9x9x55张128x128图片（论文说它划分了3w张），并根据论文划分数据集（SR三分之一，rDL三分之二）；pku数据无法读取，rDL提供的读取代码不适用，**已解决，手工转换为tif文件**；

- 2024.02.27:重新按比例划分BioSR数据,完成数据集rDL_BioSR/rDL_SIM_separate_v1;**疑问：不同光照强度的相同图片被分开到train和val会影响吗?**；模型对于复杂区域的SR效果降低明显；

- 2024.02.28:比较tf版与pt版的SR效果，MT样例的SSIM在0.9.PSNR在20左右；尝试构建mrc文件以比较局部图片中tf与pt的差距，由于header等问题，失败；BioSR数据局部比较中通过官方提供的权重微调的模型与普通比较，SSIM在0.68左右；

- 2024.02.29:BioSR数据局部比较中，使用我们数据训练的tf模型与GT的SSIM在0.64，与pt模型SSIM在0.9左右（Cell01-Level01-Crop7），总结与官方权重差距产生在数据集大小及挑选上（官方使用总计3w张进行挑选的，我们划分了5k张左右且不进行挑选）；BioSR_separatev2，DFCANmodel简单修改；新建Moiré_Generate，提前生成仿真图片存放，加快训练速度（**搁置，光学信息存疑**），tf版本中训练光照从GT中获得，Pre中光照由训练数据的平均获得；

- 2024.03.04:改变rDL_tf版的光学来源，从gt改为input，**影响微乎其微**；

- 2024.03.08:存储val数据，后续可视化

- 2024.03.15:**rDL的Simple中9通道只取一个通道进行推理及指标计算，并且输出的**;修改cal_comp中单独计算每个通道SSIM、PSNR等为整体统一计算;**from skimage.metrics import structural_similarity as compare_ssim 于pytorch_ssim库得出的值不同，后者的较低，**;仍存在很大出入;

- 2024.03.16:rDL在预测中采用successive noise raw叠加，增强其中的条纹信息来辅助估计，**测试数据统计中暂时不使用，一张张预测**；

- 2024.03.20:处理PKU数据到模型所需格式，共49 * 55 * 11 张图片，但是PKU数据集数据质量不佳；修改代码val和sample中的SSIM计算函数，保持与测试中一致，使用Skimage库；开始使用PKU数据进行训练；

- 2024.03.21:根据实验结果表明，**光照信息OTF对于PKU数据集的影响几乎没有**；创建Pattern_image分支，尝试将Pattern-moduled images的生成分离，提高训练效率；创建rDL_NoPattern模型，取消SIM模拟过程，将SR_Inference输出的图片通过一个卷积层进行维度改变后，直接送入MPE中，进行实验结果比较；

- 2024.03.26:修改rDL_Denoise_NoPattern模型及训练代码，提供encoder分支MPE_input_channel数目设置选择（3/9），3对应与原模型训练方法，9需要增加mid_channel数目提高模型容量；

- 2024.03.27:新建rDL_NoPatternv2_test用于测试NoPatternv2模型，NoPattern模型已被替代，若要测试需要从github上参考下载模型；