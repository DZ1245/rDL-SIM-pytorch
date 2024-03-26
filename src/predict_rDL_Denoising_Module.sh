CUDA_VISIBLE_DEVICES='2' CUDA_DEVICE_ORDER=PCI_BUS_ID \
python  -m torch.distributed.launch --nproc_per_node 1 \
                                    predict_rDL_Denoising_Module.py \
                                    --data_folder Microtubules \
                                    --dataset Microtubules \
                                    --input_height 512 --input_width 512 \
                                    --SR_model_name DFCAN --DN_model_name rDL_Denoiser \
                                    --load_weights_flag 1 --Encoder_type MPE+PFE \
                                    --exp_name rDL_pt_PKUse_onlyPFE --resume_name rDL_pt_PKUse_onlyPFE \
                                    --SR_resume_name DFCAN_PKU_separate \ 