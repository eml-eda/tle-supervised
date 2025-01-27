CUDA_VISIBLE_DEVICES=2,3,
# (
#     python3 pretrain_all_datasets.py \
#         --device 3 \
#         --model mae \
#         --dir1 /space/benfenati/data_folder/SHM/AnomalyDetection_SS335/ \
#         --dir2 /space/benfenati/data_folder/SHM/Vehicles_Roccaprebalza/ \
#         --dir3 /space/benfenati/data_folder/SHM/Vehicles_Sacertis/ \
#         --window_size 1190
# ) | tee logs/pretrain_all_datasets_mae.log

(
    python3 pretrain_all_datasets.py \
        --device 3 \
        --model tcn \
        --dir1 /space/benfenati/data_folder/SHM/AnomalyDetection_SS335/ \
        --dir2 /space/benfenati/data_folder/SHM/Vehicles_Roccaprebalza/ \
        --dir3 /space/benfenati/data_folder/SHM/Vehicles_Sacertis/ \
        --window_size 1190
) | tee logs/pretrain_all_datasets_tcn.log

(
    python3 pretrain_all_datasets.py \
        --device 3 \
        --model lstm \
        --dir1 /space/benfenati/data_folder/SHM/AnomalyDetection_SS335/ \
        --dir2 /space/benfenati/data_folder/SHM/Vehicles_Roccaprebalza/ \
        --dir3 /space/benfenati/data_folder/SHM/Vehicles_Sacertis/ \
        --window_size 1190
) | tee logs/pretrain_all_datasets_lstm.log