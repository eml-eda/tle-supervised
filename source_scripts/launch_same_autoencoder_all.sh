#!/usr/bin/env bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd ..

gpu=$1

if [[ $gpu == "0" ]]; then
    echo "Run on GPU 0"
    export CUDA_VISIBLE_DEVICES=0
fi
if [[ $gpu == "1" ]]; then
    echo "Run on GPU 1"
    export CUDA_VISIBLE_DEVICES=1
fi
if [[ $gpu == "2" ]]; then
    echo "Run on GPU 2"
    export CUDA_VISIBLE_DEVICES=2
fi
if [[ $gpu == "3" ]]; then
    echo "Run on GPU 3"
    export CUDA_VISIBLE_DEVICES=3
fi

python same_encoder_different_decoders.py --dir /baltic/users/shm_mon/SHM_Datasets_2023/Datasets/Vehicles_Roccaprebalza/ --car y_camion --task Roccaprebalza --train Yes --tail Yes --modified No | tee logs/Roccaprebalza_camion_freeze_tail.log
python same_encoder_different_decoders.py --dir /baltic/users/shm_mon/SHM_Datasets_2023/Datasets/Vehicles_Roccaprebalza/ --car y_car --task Roccaprebalza --train Yes --tail Yes --modified No | tee logs/Roccaprebalza_car_freeze_tail.log
python same_encoder_different_decoders.py --dir /baltic/users/shm_mon/SHM_Datasets_2023/Datasets/Vehicles_Roccaprebalza/ --car y_car --task Roccaprebalza --train Yes --tail Yes --modified Yes | tee logs/Roccaprebalza_car_freeze_tail_modified.log
python same_encoder_different_decoders.py --dir /baltic/users/shm_mon/SHM_Datasets_2023/Datasets/Vehicles_Roccaprebalza/ --car y_camion --task Roccaprebalza --train Yes --tail Yes --modified Yes | tee logs/Roccaprebalza_camion_freeze_tail_modified.log
python same_encoder_different_decoders.py --dir /baltic/users/shm_mon/SHM_Datasets_2023/Datasets/Vehicles_Sacertis/ --car y_car --task Sacertis --train Yes --tail Yes --modified Yes | tee logs/Sacertis_freeze_tail_modified.log
python same_encoder_different_decoders.py --dir /baltic/users/shm_mon/SHM_Datasets_2023/Datasets/Vehicles_Sacertis/ --car y_car --task Sacertis --train Yes --tail Yes --modified No | tee logs/Sacertis_freeze_tail.log
