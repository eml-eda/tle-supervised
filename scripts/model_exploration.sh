CUDA_VISIBLE_DEVICES=3,
(
    python model_exploration.py --device 3
) | tee logs/model_exploration.log