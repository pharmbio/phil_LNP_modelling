Python scripts and jupyter notebooks to accompany the manuscript:

**Deep learning models for lipid-nanoparticle-based drug delivery**

Authors: Philip J Harrison, Håkan Wieslander, Alan Sabirsh, Johan Karlsson, Victor Malmsjö, Andreas Hellander, Carolina Wählby and Ola Spjuth.

Note: The LNP data used in the scripts and notebooks is not included in this repository.

**1. LNP_CNN_data_prep.ipynb**

Data preparation to extract the cell-level time-lapse data needed for the CNNs.

**2. LNP_CNN_train.ipynb**

Training the CNN between time points 1 and 20 in two prediction models (classification and regression) and performing 5-fold cross-validation.

**3. LNP_time-series_data_prep.ipynb**

Using the trained CNNs create the time-series data required for the LSTM and tsfresh based applications.

**4. LNP_LSTM_model_selection.ipynb**

200 sample grid search for the best LSTM model arcitecture for each prediction mode and cross-validation fold.

**5. LNP_LSTM_train.ipynb**

Train the best LSTM models from the model selection and save out predictions on the test set.


**6. LNP_tsfresh_efficient_extract_select_PCA.py**

Using tsfresh with the "efficient parameters" setting extract and select the relevant time-series features, followed by PCA for dimenion reduction.

**7. LNP_tsfresh_efficient_gbm.ipynb**

Gradient boosting machine (GBM) grid search based on the time series features derived from (6) and save out predictions on the tests set from the best GBM model.
