
CUDA_VISIBLE_DEVICES=2,3,
(
    python3 uc2.py \
        --device 2 \
        --dir /space/benfenati/data_folder/SHM/Vehicles_Roccaprebalza/ \
        --model lstm \
        --car y_car \
) | tee logs/uc2_lstm_y_car.log

(
    python3 uc2.py \
        --device 2 \
        --dir /space/benfenati/data_folder/SHM/Vehicles_Roccaprebalza/ \
        --model lstm \
        --car y_camion \
) | tee logs/uc2_lstm_y_camion.log

