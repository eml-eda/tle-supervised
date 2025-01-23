
(
    python3 uc1.py \
        --device 2 \
        --dir /space/benfenati/data_folder/SHM/AnomalyDetection_SS335/ \
        --model soa \
        --window_size 490
) | tee logs/uc1_soa.log

(
    python3 uc1.py \
        --device 2 \
        --dir /space/benfenati/data_folder/SHM/AnomalyDetection_SS335/ \
        --model autoencoder \
        --window_size 490
) | tee logs/uc1_autoencoder.log 
 
