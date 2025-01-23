
(
    python3 uc2.py \
        --device 3 \
        --dir /space/benfenati/data_folder/SHM/Vehicles_Roccaprebalza/ \
        --model tcn \
        --car y_car \
) | tee logs/uc2_tcn_y_car.log

(
    python3 uc2.py \
        --device 3 \
        --dir /space/benfenati/data_folder/SHM/Vehicles_Roccaprebalza/ \
        --model tcn \
        --car y_camion \
) | tee logs/uc2_tcn_y_camion.log

