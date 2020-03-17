from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from scipy.stats import uniform, randint

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters, MinimalFCParameters, settings

# set the logger to error level
# tsfresh outputs many warnings for features that cannot be calculated
import logging
logging.basicConfig(level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# from tsfresh github examples
# note in this function they write normalize where they should really have written standardize!
class PCAForPandas(PCA):
    """This class is just a small wrapper around the PCA estimator of sklearn 
    including normalization to make it compatible with pandas DataFrames."""

    def __init__(self, **kwargs):
        self._z_scaler = StandardScaler()
        super(self.__class__, self).__init__(**kwargs)

        self._X_columns = None

    def fit(self, X, y=None):
        """Normalize X and call the fit method of the base class with numpy arrays
        instead of pandas data frames."""

        X = self._prepare(X)

        self._z_scaler.fit(X.values, y)
        z_data = self._z_scaler.transform(X.values, y)

        return super(self.__class__, self).fit(z_data, y)

    def fit_transform(self, X, y=None):
        """Call the fit and the transform method of this class."""

        X = self._prepare(X)

        self.fit(X, y)
        return self.transform(X, y)

    def transform(self, X, y=None):
        """Normalize X and call the transform method of the base class with numpy arrays
        instead of pandas data frames."""

        X = self._prepare(X)

        z_data = self._z_scaler.transform(X.values, y)

        transformed_ndarray = super(self.__class__, self).transform(z_data)

        pandas_df = pd.DataFrame(transformed_ndarray)
        pandas_df.columns = ["pca_{}".format(i) for i in range(len(pandas_df.columns))]

        return pandas_df

    def _prepare(self, X):
        """Check if the data is a pandas DataFrame and sorts the column names.

        :raise AttributeError: if pandas is not a DataFrame or the columns of the new X 
        is not compatible with the columns from the previous X data"""
        
        if not isinstance(X, pd.DataFrame):
            raise AttributeError("X is not a pandas DataFrame")

        X.sort_index(axis=1, inplace=True)

        if self._X_columns is not None:
            if self._X_columns != list(X.columns):
                raise AttributeError("The columns of the new X is not compatible with the columns from the previous X data")
        else:
            self._X_columns = list(X.columns)

        return X

n_time = 20
n_latent = 32

fold = 'fold5'
fit_method = 'regress'

train_ts = np.load('/scratch-shared/phil/LNP/LNP_data_09/train_ts_' + fit_method + '_' + fold + '.npy')
train_y = np.load('/scratch-shared/phil/LNP/LNP_data_09/train_cell_gfp_' + fit_method + '_' + fold + '.npy')
train_ids = np.load('/scratch-shared/phil/LNP/LNP_data_09/train_cell_ids_' + fit_method + '_' + fold + '.npy')

test_ts = np.load('/scratch-shared/phil/LNP/LNP_data_09/test_ts_' + fit_method + '_' + fold + '.npy')
test_y = np.load('/scratch-shared/phil/LNP/LNP_data_09/test_cell_gfp_' + fit_method + '_' + fold + '.npy')
test_ids = np.load('/scratch-shared/phil/LNP/LNP_data_09/test_cell_ids_' + fit_method + '_' + fold + '.npy')

train_index = []
valid_index = []

for i in range(len(train_ids)):
    s0 = train_ids[i].split('train/')
    s1 = s0[1].split('_')[0]
    if s1 == fold:
        valid_index.append(i)
    else:
        train_index.append(i)

train_ts_train = train_ts[train_index]
train_y_train = train_y[train_index]

col_names = ['id', 'time']
for i in range(10):
    col_names.append('x0' + str(i))
for i in range(10, n_latent):
    col_names.append('x' + str(i))

X0 = np.ndarray((len(train_ts_train) * n_time, n_latent + 2))
row_index = 0
for i in range(len(train_ts_train)):
    for j in range(n_time):
        X0[row_index, 0] = i
        X0[row_index, 1] = j
        X0[row_index, 2:34] = train_ts_train[i, j, :]
        row_index += 1

X0 = pd.DataFrame(X0, columns=col_names)
X0['id'] = X0['id'].astype(int)
X0['time'] = X0['time'].astype(int)

# across all training data, including validation set
X0_train = np.ndarray((len(train_ts) * n_time, n_latent + 2))
row_index = 0
for i in range(len(train_ts)):
    for j in range(n_time):
        X0_train[row_index, 0] = i
        X0_train[row_index, 1] = j
        X0_train[row_index, 2:34] = train_ts[i, j, :]
        row_index += 1

X0_train = pd.DataFrame(X0_train, columns=col_names)
X0_train['id'] = X0_train['id'].astype(int)
X0_train['time'] = X0_train['time'].astype(int)

# and for the test data
X0_test = np.ndarray((len(test_ts) * n_time, n_latent + 2))
row_index = 0
for i in range(len(test_ts)):
    for j in range(n_time):
        X0_test[row_index, 0] = i
        X0_test[row_index, 1] = j
        X0_test[row_index, 2:34] = test_ts[i, j, :]
        row_index += 1

X0_test = pd.DataFrame(X0_test, columns=col_names)
X0_test['id'] = X0_test['id'].astype(int)
X0_test['time'] = X0_test['time'].astype(int)

print('extracting training (train) features')
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
extraction_settings = EfficientFCParameters() # for tsfresh
X = extract_features(X0,
                     column_id='id', column_sort='time',
                     default_fc_parameters=extraction_settings,
                     impute_function= impute,
                     n_jobs=3)

print('selecting training features')
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
X_filtered = select_features(X, train_y_train, n_jobs=3)
print('X_filtered shape=')
print(X_filtered.shape)

print('extracting selected features for ALL training data')
X_train_filtered = extract_features(X0_train, column_id='id', column_sort='time',
                                    kind_to_fc_parameters=settings.from_columns(X_filtered.columns),
                                    impute_function=impute, n_jobs=3)

print('X_train_filtered shape=')
print(X_train_filtered.shape)


print('extracting selected features for test data')
X_test_filtered = extract_features(X0_test, column_id='id', column_sort='time',
                                   kind_to_fc_parameters=settings.from_columns(X_filtered.columns),
                                   impute_function=impute, n_jobs=3)

print('X_test_filtered shape=')
print(X_test_filtered.shape)

print('running PCA')
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
pca_train = PCAForPandas(n_components=n_latent)
X_train_pca = pca_train.fit_transform(X_train_filtered)
X_test_pca = pca_train.transform(X_test_filtered)


X_train_pca.to_csv('/scratch-shared/phil/LNP/LNP_data_09/tsfresh_efficient_pca_train_features_' + fit_method + '_' + fold + '.csv')
X_test_pca.to_csv('/scratch-shared/phil/LNP/LNP_data_09/tsfresh_efficient_pca_test_features_' + fit_method + '_' + fold + '.csv')

print('FINISHED!')
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
