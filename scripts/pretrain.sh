
(
    python pretrain_all_datasets.py \
        --device 3 \
        --dir1 /space/benfenati/data_folder/SHM/AnomalyDetection_SS335/ \
        --dir2 /space/benfenati/data_folder/SHM/Vehicles_Roccaprebalza/ \
        --dir3 /space/benfenati/data_folder/SHM/Vehicles_Sacertis/ \
        --window_size 1190
) | tee logs/pretrain_all_datasets.log