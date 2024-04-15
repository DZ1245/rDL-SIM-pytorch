CUDA_VISIBLE_DEVICES='4,5' CUDA_DEVICE_ORDER=PCI_BUS_ID \
python  -m torch.distributed.launch --nproc_per_node 2 --master_port 29771 \
                                    train_rDL_Denoising_Module_NoPattern.py \
                                    --root_path /mnt/data2_16T/datasets/zhi.deng/Biology_cell/rDL_SIM_data/rDL_pku/rDL_SIM_separate/DN/ \
                                    --data_folder Microtubules \
                                    --dataset Microtubules \
                                    --input_height 128 --input_width 128\
                                    --SR_model_name DFCAN --DN_model_name rDL_Denoiser \
                                    --DN_attention_mode SEnet --Encoder_type MPE+PFE \
                                    --total_epoch 10000 --sample_epoch 2 --log_iter 100 \
                                    --start_lr 1e-4 --lr_decay_factor 0.5 --ssim_weight 1e-1 \
                                    --num_gpu 1 --num_workers 15 --norm_flag 1 --load_weights_flag 0 \
                                    --exp_name rDL_PKUse_NoPatternv2 \
                                    --resume_name rDL_PKUse_NoPatternv2 \
                                    --SR_resume_name DFCAN_PKU_separate \
                                    --batch_size 3 --MPE_input_channel 3 \

# --Encoder_type MPE+PFE