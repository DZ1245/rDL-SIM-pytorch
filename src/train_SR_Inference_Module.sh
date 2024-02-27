CUDA_VISIBLE_DEVICES='0,1,2,3' CUDA_DEVICE_ORDER=PCI_BUS_ID \
python -m torch.distributed.launch  --nproc_per_node 4 \
                                    train_SR_Inference_Module.py \
                                    --root_path /mnt/data2_16T/datasets/zhi.deng/Biology_cell/rDL_SIM_data/rDL_BioSR/rDL_SIM_separate_v1/SR \
                                    --data_folder Microtubules \
                                    --dataset Microtubules \
                                    --input_height 128 --input_width 128 \
                                    --input_channels 9 --out_channels 1 --scale_factor 2 \
                                    --model_name DFCAN \
                                    --norm_flag 1 --total_epoch 10000 --sample_epoch 2 \
                                    --log_iter 10 --num_workers 30 \
                                    --batch_size 28 --start_lr 1e-4 --lr_decay_factor 0.5 \
                                    --ssim_weight 1e-1 \
                                    --load_weights_flag 0 \
                                    --exp_name BioSR_separate \
                                    --resume_name BioSR_separate \ 

                                        
# 4090 DFCAN batchsize = 28 DFCAN_SimAM batchsize = 21
# --root_path /mnt/data2_16T/datasets/zhi.deng/Biology_cell/rDL_SIM/SR/ \
# 2080ti 10 
# --root_path /data/home/dz/rDL_SIM/SR/ \
