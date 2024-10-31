## About datasets used in the `Chem_ML` notebook

The `small_set.txt` is a random selected subset from the OQMD. The dataset is used for testing the model. The full dataset has not yet been used since my personal computer does not have enough memory to load the full dataset. Currently I am working on migrating the code to run on Google Colab to use their GPU and more memory.

The `oqmd.csv` is the full dataset from the OQMD. The dataset contains about 500k data points.

The `features_OQMD_100k.npy` and `targets_OQMD_100k.npy` are the preprocessed features and targets of the 100k OQMD dataset. The 100k dataset is a subset of the full dataset. The 100k dataset is used for training the model in the `Chem_ML_OQMD_100k.ipynb` notebook.