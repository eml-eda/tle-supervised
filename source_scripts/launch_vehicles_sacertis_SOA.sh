#!/usr/bin/env bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd ..

python vehicles_sacertis_example.py --dir /baltic/users/shm_mon/SHM_Datasets_2023/Datasets/Vehicles_Sacertis/ --model SOA | tee logs/Vehicles_Sacertis_SOA.log
