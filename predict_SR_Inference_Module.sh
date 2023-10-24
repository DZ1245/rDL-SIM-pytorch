CUDA_VISIBLE_DEVICES='0' CUDA_DEVICE_ORDER=PCI_BUS_ID \
python src/predict_SR_Inference_Module.py --root_path /mnt/data2_16T/datasets/zhi.deng/Biology_cell/rDL_SIM/SR/ \
                                          --data_folder Microtubules \
                                          --dataset Microtubules \
                                          --model_name DFCAN \
                                          --load_weights_flag 1 --norm_flag 1 --num_gpu 1 \
                                          --exp_name Model_noCast --resume_name Model_noCast \

