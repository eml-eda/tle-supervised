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

python vehicles_roccaprebalza_example.py --dir /baltic/users/shm_mon/SHM_Datasets_2023/Datasets/Vehicles_Roccaprebalza/ --model autoencoder --car y_car | tee logs/Vehicles_Roccaprebalza_y_car_autoencoder.log
python vehicles_roccaprebalza_example.py --dir /baltic/users/shm_mon/SHM_Datasets_2023/Datasets/Vehicles_Roccaprebalza/ --model autoencoder --car y_camion | tee logs/Vehicles_Roccaprebalza_y_camion_autoencoder.log
