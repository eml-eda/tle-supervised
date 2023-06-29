import numpy as np 
import sklearn.preprocessing

mse_w = lambda x, xr: np.mean((x - xr)**2, axis=0)
mse = lambda x, xr: np.mean(mse_w(x, xr))

class pca_class:
    def __init__(
        self, 
        input_dim, 
        CF):
        """
        Principal Component Analysis builder and predictor. 

        :param input_dim:			Input Dimension of each rolled windows 
        :param CF:					Latent dimension of subspace domain
        """
        self.cf = CF
        self.dim = input_dim 

    def fit(self, dataset):
        """
        fit the pca to find eigen values and eigen vectors of training set. 

        :param dataset: trainset in shape M (number of features) x N (number of samples)
        :retrun Ex: 	Eigen-Values of PCA 
        :return Vx: 	Eigen-Vectors of PCA 
        :return k:		Optimal Sub-space domain 
        """
        energy_percentage = 0.97
        Cx = np.dot(dataset.T, dataset)/self.dim 
        Ex, Vx = np.linalg.eigh(Cx)
        Ex = Ex[::-1]
        Vx = Vx[:, ::-1]
        lcumsum = np.cumsum(Ex/np.sum(Ex))
        k = np.argmax(lcumsum > energy_percentage)
        return Ex,Vx,k

    def predict(self, ds, Vx):
        """
        Returns MSE values of interval of 15 minutes 

        :param ds:			dataset of 5 days for testing 
        :param Vx:			Trained Eigen Vectors 
        :param ds_start:	Start date in dataset
        :param ds_end: 		Finish date in dateset
        :return output:		Dictionary with 2 <value/pair> of MSE/starting date of the interval  
        """
        output = {}
        conv = (1*2.5)*2**-15
        interval = 30
        for values in np.arange(0, ds.shape[0], interval): #windows of 30 seconds
            section = ds[values:(values+interval),:]
            if section.shape[0]>0: 
                section = sklearn.preprocessing.scale(section, axis=1, with_mean=True, with_std=False, copy=False)
                x_recons = np.linalg.multi_dot([Vx[:, :self.cf], Vx[:, :self.cf].T, section.T])
                mse_temp =  mse(section, x_recons.T)
                if output == {}:
                    output['mse'] = mse_temp
                else: 
                    output['mse'] = np.append(output['mse'], mse_temp)
        return output