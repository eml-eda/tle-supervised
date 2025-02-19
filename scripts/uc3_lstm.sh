
CUDA_VISIBLE_DEVICES=2,
(
    python3 uc3.py \
        --device 2 \
        --dir /space/benfenati/data_folder/SHM/Vehicles_Sacertis/ \
        --model lstm \
        --no_pretrain True
) | tee logs/uc3_lstm_no_pretrain.log

CUDA_VISIBLE_DEVICES=2,
(
    python3 uc3.py \
        --device 2 \
        --dir /space/benfenati/data_folder/SHM/Vehicles_Sacertis/ \
        --model lstm \
        --pretrain_all True
) | tee logs/uc3_lstm_pretrain.log