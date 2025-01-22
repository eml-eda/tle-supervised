
(
    python uc2.py \
        --device 2 \
        --dir /space/benfenati/data_folder/SHM/Vehicles_Sacertis/ \
        --model soa \
) | tee logs/uc3_soa.log


(
    python uc2.py \
        --device 2 \
        --dir /space/benfenati/data_folder/SHM/Vehicles_Roccaprebalza/ \
        --model autoencoder \
        --pretrain_all
) | tee logs/uc3_autoencoder.log 
