#!/usr/bin/env bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd ..

python anomaly_detection_ss335_example.py --dir /baltic/users/shm_mon/SHM_Datasets_2023/Datasets/AnomalyDetection_SS335/ --model SOA --window_size 490 | tee logs/anomaly_detection_490_PCA.log
