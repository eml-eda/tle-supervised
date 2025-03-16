# import packages
import torch
import time
import numpy as np
import argparse
import sys

from models.pca import pca_class
from models.models_audio_mae import audioMae_vit_base
from models.models_audio_mae_regression import audioMae_vit_base_R
from models.models_tcn_regression import tcn_regression as tcn_regression_mae
from models.models_lstm_regression import lstm_regression as lstm_regression_mae
from models.soa import TorchStandardScaler, TorchSVR, TorchDecisionTree, TorchMLP, TorchKNN, TorchLinearRegression, TorchPipeline

import warnings
warnings.filterwarnings("ignore")

def run_evaluation(ntests=10):
    """Run the evaluation with specified number of tests"""
    # params for the simulation
    embed_dim = 768
    decoder_embed_dim = 512

    # define models
    # UC1 
    pca = pca_class(input_dim=1190, CF = 32)
    dummy_Vx = np.random.rand(1190, 1190)
    model_uc1 = audioMae_vit_base(embed_dim=embed_dim, decoder_embed_dim=decoder_embed_dim, norm_pix_loss=False).half().to('cuda').eval()

    # UC2/3
    steps_svr = [('scaler', TorchStandardScaler()), ('model', TorchSVR(C=10, epsilon=0.1))]
    steps_DecisionTreeRegressor = [('scaler', TorchStandardScaler()), ('model', TorchDecisionTree(max_depth=200))]
    steps_MLPRegressor = [('scaler', TorchStandardScaler()), ('model', TorchMLP())]
    steps_KNeighborsRegressor = [('scaler', TorchStandardScaler()), ('model', TorchKNN(n_neighbors=7))]
    steps_LinearRegression = [('scaler', TorchStandardScaler()), ('model', TorchLinearRegression())]
    steps = [steps_svr, steps_DecisionTreeRegressor, steps_MLPRegressor, steps_KNeighborsRegressor, steps_LinearRegression]
    names = ["SVR", "DT", "MLP", "KNN", "LR"]

    model_uc2_ours = audioMae_vit_base_R(embed_dim=embed_dim, decoder_embed_dim=decoder_embed_dim, norm_pix_loss=True, mask_ratio = 0.2).half().to('cuda').eval()
    model_tcn = tcn_regression_mae(embed_dim=embed_dim, decoder_embed_dim=decoder_embed_dim, mask_ratio = 0.2).half().to('cuda').eval()
    model = lstm_regression_mae(embed_dim=embed_dim,decoder_embed_dim=decoder_embed_dim, mask_ratio = 0.2).half().to('cuda').eval()

    ## define dummy input data
    dummy_Vx = np.random.rand(1190, 1190)
    dummy_data_pca = np.random.rand(1, 1190)
    dummy_data_ours = torch.rand(1,1,100,100, dtype=torch.float16).to('cuda')

    dummy_train = np.random.randn(609,12)
    dummy_labels = np.random.randn(609,1)
    dummy_data_soa = np.random.rand(1, 12)

    # dummy warmup
    for _ in range(5):
        model_uc1(dummy_data_ours)

    # uc1 profiling
    ## pca
    latencies = []
    for _ in range(ntests):
        start = time.time()
        pca_result_normal = pca.predict(dummy_data_pca, dummy_Vx)
        end = time.time()
        latencies.append(end - start)
    print(f'avg latency on UC1-PCA: {sum(latencies)/len(latencies):.3f} seconds')
    ## ours
    latencies = []
    for _ in range(ntests):
        start = time.time()
        loss, _, _ = model_uc1(dummy_data_ours)
        end = time.time()
        latencies.append(end - start)
    print(f'avg latency on UC1-ours: {sum(latencies)/len(latencies):.3f} seconds')

    # uc2 profiling
    ## soa
    for i, step in enumerate(steps):
        pipeline = TorchPipeline(step)
        pipeline.fit(dummy_train, dummy_labels)
        latencies = []
        for _ in range(ntests):
            start = time.time()
            y_predicted = pipeline.predict(dummy_data_soa)
            end = time.time()
            latencies.append(end - start)
        print(f'avg latency on UC2/3-{names[i]} : {sum(latencies)/len(latencies):.3f} seconds')

    names_deep = ["ours", "tcn", "lstm"]
    for i, model in enumerate([model_uc2_ours, model_tcn, model]):
        latencies = []
        for _ in range(ntests):
            start = time.time()
            loss, _ = model(dummy_data_ours)
            end = time.time()
            latencies.append(end - start)
        print(f'avg latency on UC2/3-{names_deep[i]} : {sum(latencies)/len(latencies):.3f} seconds')


def main():
    parser = argparse.ArgumentParser(description='Evaluate model latencies')
    parser.add_argument('--ntests', type=int, default=10,
                        help='Number of test runs for latency measurement (default: 10)')
    args = parser.parse_args()
    
    print(f"Running evaluation with {args.ntests} test iterations")
    run_evaluation(ntests=args.ntests)


if __name__ == "__main__":
    main()