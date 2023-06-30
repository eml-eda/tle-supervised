### Data

For data, ask to Alessio Burrello.
They are offline:
- Row data:  DataFrame__sensori_rilevazioni_granulare_acc_normalizzate.pkl
- data_60, data_50, ..etc.: day_time_60.pkl, ...  day_time_10.pkl

### Notebooks

- Dataset_creation.ipynb: scripts to create and test the dataset of different window dimensions from DataFrame__sensori_rilevazioni_granulare_acc_normalizzate.pkl. Slightly different to the one stored and generated from Giovanni.
- SVR_example_SUSCOM_paper.ipynb: example of script to reproduce the results on the paper with 50 features, on dataset with windows of 60 seconds.


## Experiments
Table 1 -- Datasets
Fig. Result1 Anomaly detection -- PCA vs Autoencoder --> error over time with average
Fig. Result2 Vehicle detection 1 -- Bar plots with SoTA + Autoencoder
Fig. Result3 Vehicle detection 2 -- Bar plots with SoTA + Autoencoder
Fig. Ablation4 Comparison with bar plots with the three tasks -- No pretraining, Normal, pretraining with all the dataset togethers
Fig. Ablation5 Bigger models and smaller models (increase/decrease number of blocks)
Table 2 -- deployment high level tipo suscom?

Anomaly detection -- Window_size --> 12 seconds  (1198 samples) Window_step --> 2 second self.th = 3.125*(10**-5) -- 5 seconds for PCA
Vehicles Roccaprebalza -- 60 seconds window size, shift 2 seconds -- sensor z10D41
STILL ERROR IN TESTING ON TRAINING ON THIS DATASET
Vehicle Sacertis -- launched with predefined pretraining, finetuning and testing times in the script, using all sensors