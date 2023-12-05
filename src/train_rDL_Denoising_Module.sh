CUDA_VISIBLE_DEVICES='3' CUDA_DEVICE_ORDER=PCI_BUS_ID \
python  -m torch.distributed.launch --nproc_per_node 1 \
                                    train_rDL_Denoising_Module.py \
                                    --root_path /mnt/data2_16T/datasets/zhi.deng/Biology_cell/rDL_SIM_data/rDL_SIM/DN/ \
                                    --data_folder Microtubules \
                                    --dataset Microtubules \
                                    --input_height 128 --input_width 128\
                                    --SR_model_name DFCAN --DN_model_name rDL_Denoiser --DN_attention_mode CBAM \
                                    --total_epoch 10000 --sample_epoch 2 --log_iter 10 \
                                    --batch_size 1 --start_lr 1e-4 --lr_decay_factor 0.5 --ssim_weight 1e-1 \
                                    --num_gpu 1 --num_workers 5 --load_weights_flag 0 \
                                    --exp_name Denoise_CBAM --resume_name Denoise_CBAM \
                                    --SR_resume_name best \
# 2080ti
# /data/home/dz/rDL_SIM/DN/
# 4090
# /mnt/data2_16T/datasets/zhi.deng/Biology_cell/rDL_SIM/DN/