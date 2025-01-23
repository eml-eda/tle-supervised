
(
    python3 uc2.py \
        --device 2 \
        --dir /space/benfenati/data_folder/SHM/Vehicles_Roccaprebalza/ \
        --model soa \
        --car y_car \
) | tee logs/uc2_soa_y_car.log

(
    python3 uc2.py \
        --device 2 \
        --dir /space/benfenati/data_folder/SHM/Vehicles_Roccaprebalza/ \
        --model soa \
        --car y_camion \
) | tee logs/uc2_soa_y_camion.log

(
    python3 uc2.py \
        --device 2 \
        --dir /space/benfenati/data_folder/SHM/Vehicles_Roccaprebalza/ \
        --model autoencoder \
        --car y_car \
        --pretrain_all
) | tee logs/uc2_autoencoder_y_car.log 
 
(
    python3 uc2.py \
        --device 2 \
        --dir /space/benfenati/data_folder/SHM/Vehicles_Roccaprebalza/ \
        --model autoencoder \
        --car y_camion \
        --pretrain_all
) | tee logs/uc2_autoencoder_y_camion.log 
