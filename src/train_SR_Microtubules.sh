CUDA_VISIBLE_DEVICES='0,1' CUDA_DEVICE_ORDER=PCI_BUS_ID \
python train_SR_Inference_Module.py --root_path /mnt/data2_16T/datasets/zhi.deng/Biology_cell/rDL_SIM/SR/ \
                                        --data_folder Microtubules \
                                        --dataset Microtubules \
                                        --input_height 128 --input_width 128 --input_channels 9 \
                                        --model_name DFCAN \
                                        --norm_flag 1 \
                                        --total_epoch 10000 --sample_epoch 2 \
                                        --log_iter 10 \
                                        --batch_size 20 --start_lr 1e-4 --lr_decay_factor 0.5 \
                                        --ssim_weight 1e-1 \
                                        --num_gpu 2 --num_workers 30\
                                        --load_weights_flag 0 \
                                        --exp_name Model_fix_tensorboard \
                                        --resume_name Model_fix_tensorboard \

                                        
# 4090 batchsize = 25 * gpu_num
# --root_path /mnt/data2_16T/datasets/zhi.deng/Biology_cell/rDL_SIM/SR/ \
# 2080ti 10 * gpu_num
# --root_path /data/home/dz/rDL_SIM/SR/ \