CUDA_VISIBLE_DEVICES='2' CUDA_DEVICE_ORDER=PCI_BUS_ID \
python  -m torch.distributed.launch --nproc_per_node 1 \
                                    predict_rDL_Denoising_Module.py \
                                    --root_path /mnt/data2_16T/datasets/zhi.deng/Biology_cell/rDL_SIM/SR/ \
                                    --data_folder Microtubules \
                                    --dataset Microtubules \
                                    --input_height 128 --input_width 128\
                                    --SR_model_name DFCAN --DN_model_name rDL_Denoiser\
                                    --num_gpu 1 --load_weights_flag 1\
                                    --exp_name Denoise --resume_name Denoise \
                                    --SR_resume_name best \