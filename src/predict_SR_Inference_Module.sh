CUDA_VISIBLE_DEVICES='2' CUDA_DEVICE_ORDER=PCI_BUS_ID \
python -m torch.distributed.launch  --nproc_per_node 1 \
                                    predict_SR_Inference_Module.py \
                                    --root_path /mnt/data2_16T/datasets/zhi.deng/Biology_cell/rDL_SIM/SR/ \
                                    --data_folder DL1to1downsample --dataset DL1to1downsample \
                                    --input_channels 1 --out_channels 1 --scale_factor 2\
                                    --model_name DFCAN \
                                    --load_weights_flag 1 --norm_flag 1 --num_gpu 1 \
                                    --exp_name DL1to1downsample_finetuneMT --resume_name DL1to1downsample_finetuneMT \

