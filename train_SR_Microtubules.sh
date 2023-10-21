python src/train_SR_Inference_Module.py --root_path /data/home/dz/rDL_SIM/SR/ \
                                        --data_folder Microtubules \
                                        --dataset Microtubules \
                                        --input_height 128 --input_width 128 --input_channels 9 \
                                        --model_name DFCAN \
                                        --total_epoch 10000 --sample_epoch 2 \
                                        --log_iter 20 \
                                        --batch_size 12 --start_lr 1e-4 --lr_decay_factor 0.5 \
                                        --ssim_weight 1e-1 \
                                        --num_gpu 2 --gpu_id '0,1' \
                                        --exp_name Test \
                                        
