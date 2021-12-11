
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

#from keras.backend import clear_session
tf.keras.backend.clear_session()

# Keras can't work with csr_matrix.
# Convert to a numpy array.
X_train = X_train.toarray()
#X_test = tf.x_test.toarray()

#input_dim = X_train.shape[1]  # Number of features
#model = Sequential()
#model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
#model.add(layers.Dense(1, activation='sigmoid'))
