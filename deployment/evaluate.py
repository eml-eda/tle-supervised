# import packages
import torch
import time

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from models.pca import pca_class
from models.models_audio_mae import audioMae_vit_base
from models.models_audio_mae_regression import audioMae_vit_base_R
from models.models_tcn_regression import tcn_regression as tcn_regression_mae
from models.models_lstm_regression import lstm_regression as lstm_regression_mae

# params for the simulation
embed_dim = 768
decoder_embed_dim = 512
ntests = 10

# define models
# UC1 
pca = pca_class(input_dim=1190, CF = 32)
model_uc1 = audioMae_vit_base(embed_dim=embed_dim, decoder_embed_dim=decoder_embed_dim, norm_pix_loss=False).half().to('cuda').eval()

# UC2/3
steps_svr = [( 'scaler', StandardScaler() ), ('svr', SVR(kernel = 'rbf',epsilon=0.1, C=10))]
steps_DecisionTreeRegressor = [( 'scaler', StandardScaler()), ('model', DecisionTreeRegressor(max_depth=200))]
steps_MLPRegressor = [('scaler', QuantileTransformer()), ('model', MLPRegressor(hidden_layer_sizes=(100,100,100)))]
steps_KNeighborsRegressor = [( 'scaler', StandardScaler() ), ('model', KNeighborsRegressor(n_neighbors=7))]
steps_BayesianRidge = [( 'scaler', StandardScaler() ), ('model', LinearRegression())]
steps = [steps_svr, steps_DecisionTreeRegressor, steps_MLPRegressor, steps_KNeighborsRegressor, steps_BayesianRidge]
names = ["SVR", "DT", "MLP", "KNN", "LR"]
model_uc2_ours = audioMae_vit_base_R(embed_dim=embed_dim, decoder_embed_dim=decoder_embed_dim, norm_pix_loss=True, mask_ratio = 0.2).half().to('cuda').eval()
model_tcn = tcn_regression_mae(embed_dim=embed_dim, decoder_embed_dim=decoder_embed_dim, mask_ratio = 0.2).half().to('cuda').eval()
model = lstm_regression_mae(embed_dim=embed_dim,decoder_embed_dim=decoder_embed_dim, mask_ratio = 0.2).half().to('cuda').eval()

## define dummy input data
data= torch.rand(1,1,100,100, dtype=torch.float16).to('cuda')

# dummy warmup
for _ in range(5):
    model_uc1(data)

# uc1 profiling
## pca
latencies = []
for _ in range(ntests):
    start = time.time()
    pca_result_normal  = pca.predict(data, Vx)
    end = time.time()
    latencies.append(end - start)
print(f'average latency on UC1-PCA: {sum(latencies)/len(latencies):.3f} seconds')
## ours
latencies = []
for _ in range(ntests):
    start = time.time()
    loss, _, _ = model_uc1(data)
    end = time.time()
    latencies.append(end - start)
print(f'average latency on UC1-ours: {sum(latencies)/len(latencies):.3f} seconds')

# uc2 profiling
## soa
for i, step in enumerate(steps):
    pipeline = Pipeline(step)
    latencies = []
    for _ in range(ntests):
        start = time.time()
        y_predicted = pipeline.predict(data)
        end = time.time()
        latencies.append(end - start)
    print(f'average latency on UC2/3-{names[i]} : {sum(latencies)/len(latencies):.3f} seconds')

names_deep = ["ours", "tcn", "lstm"]
for i, model in enumerate([model_uc2_ours, model_tcn, model]):
    latencies = []
    for _ in range(ntests):
        start = time.time()
        loss, _ = model(data)
        end = time.time()
        latencies.append(end - start)
    print(f'average latency on UC2/3-{names_deep[i]} : {sum(latencies)/len(latencies):.3f} seconds')