#!/usr/bin/env bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd ..

python vehicles_roccaprebalza_example.py --dir /baltic/users/shm_mon/SHM_Datasets_2023/Datasets/Vehicles_Roccaprebalza/ --model SOA --car y_car | tee logs/Vehicles_Roccaprebalza_y_car_SOA.log
python vehicles_roccaprebalza_example.py --dir /baltic/users/shm_mon/SHM_Datasets_2023/Datasets/Vehicles_Roccaprebalza/ --model SOA --car y_camion | tee logs/Vehicles_Roccaprebalza_y_camion_SOA.log
