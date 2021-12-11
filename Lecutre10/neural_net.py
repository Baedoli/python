
# Modeling ...
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tensorflow.keras.backend import clear_session
clear_session()

# Keras can't work with csr_matrix. Convert to a numpy array.


X_train = X_train.toarray()
#X_test = X_test.toarray()
#input_dim = X_train.shape[1]

# Number of features
#model = Sequential()
#model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
#model.add(layers.Dense(1, activation='sigmoid'))


