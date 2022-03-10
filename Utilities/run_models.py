import pickle
import numpy as np
import time
model_names = ['svr_car_15.pkl', 'DT_car_10.pkl', 'knn_car_15.pkl', 'LinearRegression_car_150.pkl', 'mlp_car_20.pkl', 'svr_car.pkl', 'DT_car.pkl', 'knn_car.pkl', 'LinearRegression_car.pkl', 'mlp_car.pkl']
X_test_list = [np.random.rand(1, 15), np.random.rand(1, 10), np.random.rand(1, 15), np.random.rand(1, 150), np.random.rand(1, 20), np.random.rand(1, 252), np.random.rand(1, 252), np.random.rand(1, 252), np.random.rand(1, 252), np.random.rand(1, 252)]
for j,model in enumerate(model_names):
	loaded_model = pickle.load(open(model, 'rb'))
	print(model)
	X_test = X_test_list[j]
	start_time = time.time()
	for i in np.arange(0,1000):
		result = loaded_model.predict(X_test)
	print("--- %s seconds ---" % (time.time() - start_time))