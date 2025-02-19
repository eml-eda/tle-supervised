
CUDA_VISIBLE_DEVICES=3,
(
    python3 uc3.py \
        --device 3 \
        --dir /space/benfenati/data_folder/SHM/Vehicles_Sacertis/ \
        --model tcn \
        --no_pretrain True
) | tee logs/uc3_tcn_no_pretrain.log

CUDA_VISIBLE_DEVICES=3,
(
    python3 uc3.py \
        --device 3 \
        --dir /space/benfenati/data_folder/SHM/Vehicles_Sacertis/ \
        --model tcn \
        --pretrain_all True
) | tee logs/uc3_tcn_pretrain.log