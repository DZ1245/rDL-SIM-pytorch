CUDA_VISIBLE_DEVICES='7' CUDA_DEVICE_ORDER=PCI_BUS_ID \
python  -m torch.distributed.launch --nproc_per_node 1 \
                                    predict_rDL_Denoising_Module.py \
                                    --root_path /mnt/data2_16T/datasets/zhi.deng/Biology_cell/rDL_SIM/SR/ \
                                    --data_folder Microtubules \
                                    --dataset Microtubules \
                                    --input_height 512 --input_width 512 \
                                    --SR_model_name DFCAN --DN_model_name rDL_Denoiser\
                                    --num_gpu 1 --load_weights_flag 1\
                                    --exp_name rDL_BioSR_separatev2 --resume_name rDL_BioSR_separatev2 \
                                    --SR_resume_name BioSR_separate \