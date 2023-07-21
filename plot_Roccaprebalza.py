import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.stats as stats

'''
Maintenance intervention day: 9th
Train: 24/05, 25/05, 26/05, 27/05
Validation: 28/05, 29/05, 30/05
Test post intervention: 01/06, 02/06, 03/06, 04/06 Test pre intervention: 01/05, 02/05, 03/05, 04/05

Data used from Amir:
# Week of Anomaly for test - 17 to 23 of April 2019
# Week of Normal for test - 10 to 14 of May 2019 
# Week of Normal for training - 20 or 22 to 29 of May 2019
'''

def plot_ci_manual(t, s_err, n, x, x2, y2, col, ax):
    """Return an axes of confidence bands using a simple approach.

    Notes
    -----
    .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
    .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}

    References
    ----------
    .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
       http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb

    """

    ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    ax.fill_between(x2, y2 + ci, y2 - ci, color=col, alpha=0.2)


def plot_results_SOA():
    vehicle = 'camion_'
    fig, ax = plt.subplots(2, 5, figsize= (25,10),sharey=True)
    methods = ['SVR',  'DecisionTreeRegressor',  'MLPRegressor', 'KNeighborsRegressor', 'LinearRegression']
    vehicle = 'camion_'
    base_string = './Results/Roccaprebalza_'
    names = ["SVR", "DT", "MLP", "KNN", "LR"]
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    # Modeling with Numpy
    def equation(a, b):
        """Return a 1D polynomial."""
        return np.polyval(a, b) 
    for i,algorithm in enumerate(methods):
        dataset = pd.read_csv(base_string+names[i]+'_y_camion'+'.csv')
        d_int = np.round(np.asarray(dataset.values))
        d_float = (np.asarray(dataset.values))
        x = d_float[:,0]
        y = d_float[:,1]
        ax[0,i].scatter(x,y, label=algorithm,c=colors[i],s=40, edgecolor='k',linewidths =0.5)
        p, cov = np.polyfit(x, y, 1, cov=True)                     # parameters and covariance from of the fit of 1-D polynom.
        y_model = equation(p, x)                                   # model using the fit parameters; NOTE: parameters here are coefficients

        # Statistics
        n = y.size                                           # number of observations
        m = p.size                                                 # number of parameters
        dof = n - m                                                # degrees of freedom
        t = stats.t.ppf(0.995, n - m)                              # used for CI and PI bands
        
        x2 = np.linspace(np.min(x), np.max(x), 100)
        y2 = equation(p, x2)

        # Estimates of Error in Data/Model
        resid = y - y_model                           
        chi2 = np.sum((resid / y_model)**2)                        # chi-squared; estimates error in data
        chi2_red = chi2 / dof                                      # reduced chi-squared; measures goodness of fit
        s_err = np.sqrt(np.sum(resid**2) / dof)                    # standard deviation of the error
        ax[0,i].plot(x, y_model, color=colors[i], linewidth=1.5, alpha=0.5)  
        plot_ci_manual(t, s_err, n, x, x2, y2, colors[i], ax[0,i])

        ax[0,i].plot([0,10],[0,10], color = 'k')
        ax[0,i].axis([0,10,0,10])
        ax[0,i].legend()

    for i,algorithm in enumerate(methods):
        dataset = pd.read_csv(base_string+names[i]+'_y_car'+'.csv')
        d_int = np.round(np.asarray(dataset.values))
        d_float = (np.asarray(dataset.values))
        x = d_float[:,0]
        y = d_float[:,1]
        ax[1,i].scatter(x,y, label=algorithm,c=colors[i],s=40, edgecolor='k',linewidths =0.5)
        p, cov = np.polyfit(x, y, 1, cov=True)                     # parameters and covariance from of the fit of 1-D polynom.
        y_model = equation(p, x)                                   # model using the fit parameters; NOTE: parameters here are coefficients

        # Statistics
        n = y.size                                           # number of observations
        m = p.size                                                 # number of parameters
        dof = n - m                                                # degrees of freedom
        t = stats.t.ppf(0.995, n - m)                              # used for CI and PI bands
        
        x2 = np.linspace(np.min(x), np.max(x), 100)
        y2 = equation(p, x2)

        # Estimates of Error in Data/Model
        resid = y - y_model                           
        chi2 = np.sum((resid / y_model)**2)                        # chi-squared; estimates error in data
        chi2_red = chi2 / dof                                      # reduced chi-squared; measures goodness of fit
        s_err = np.sqrt(np.sum(resid**2) / dof)                    # standard deviation of the error
        ax[1,i].plot(x, y_model, color=colors[i], linewidth=1.5, alpha=0.5)  
        plot_ci_manual(t, s_err, n, x, x2, y2, colors[i], ax[1,i])

        ax[1,i].plot([0,10],[0,10], color = 'k')
        ax[1,i].axis([0,10,0,10])
        ax[1,i].legend()

    plt.savefig('Results/images/Roccaprebalza_regression.png')

def plot_results_autoencoder():
    fig, ax = plt.subplots(1, 2, figsize= (8,4))#,sharey=True)
    methods = ['Heavy Vehicles', 'Light Vehicles']
    files = ['y_camion', 'y_car']
    base_string = './Results/csv/Roccaprebalza_'
    colors = ['g', 'k']

    # Modeling with Numpy
    def equation(a, b):
        """Return a 1D polynomial."""
        return np.polyval(a, b)

    for i, lab in enumerate(methods):
        dataset = pd.read_csv(base_string+'autoencoder_'+files[i]+'.csv')
        d_int = np.round(np.asarray(dataset.values))
        d_float = (np.asarray(dataset.values))
        x = d_float[:,0]
        y = d_float[:,1]
        ax[i].scatter(x,y, label=lab,c=colors[i],s=40, edgecolor='k',linewidths =0.5)
        p, cov = np.polyfit(x, y, 1, cov=True)                     # parameters and covariance from of the fit of 1-D polynom.
        y_model = equation(p, x)                                   # model using the fit parameters; NOTE: parameters here are coefficients

        # Statistics
        n = y.size                                           # number of observations
        m = p.size                                                 # number of parameters
        dof = n - m                                                # degrees of freedom
        t = stats.t.ppf(0.995, n - m)                              # used for CI and PI bands
        
        x2 = np.linspace(np.min(x), np.max(x), 100)
        y2 = equation(p, x2)

        # Estimates of Error in Data/Model
        resid = y - y_model                           
        chi2 = np.sum((resid / y_model)**2)                        # chi-squared; estimates error in data
        chi2_red = chi2 / dof                                      # reduced chi-squared; measures goodness of fit
        s_err = np.sqrt(np.sum(resid**2) / dof)                    # standard deviation of the error
        ax[i].plot(x, y_model, color=colors[i], linewidth=1.5, alpha=0.5)  
        plot_ci_manual(t, s_err, n, x, x2, y2, colors[i], ax[i])
        if i == 0:
            ax[i].plot([0,10],[0,10], color = 'k')
        else:
            ax[i].plot([0,18],[0,18], color = 'k')
        ax[i].set_xlabel("True V.", fontsize = 12)
        ax[i].set_ylabel("Predicted V.", fontsize = 12)
        # ax[i].axis([0,10,0,10])
        ax[i].legend()
        ax[i].yaxis.grid(True)
        ax[i].tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig('Results/images/Roccaprebalza_regression_autoencoder.png')

if __name__ == "__main__":
    plot_results_autoencoder()