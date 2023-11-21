CUDA_VISIBLE_DEVICES='0' CUDA_DEVICE_ORDER=PCI_BUS_ID \
python -m torch.distributed.launch  --nproc_per_node 1 \
                                    predict_SR_Inference_Module.py \
                                    --root_path /mnt/data2_16T/datasets/zhi.deng/Biology_cell/rDL_SIM/SR/ \
                                    --data_folder MT1to1 --dataset MT1to1 \
                                    --input_channels 1 --out_channels 1 --scale_factor 2\
                                    --model_name DFCAN \
                                    --load_weights_flag 1 --norm_flag 1 --num_gpu 1 \
                                    --exp_name MT1to1_test --resume_name MT1to1_test \

