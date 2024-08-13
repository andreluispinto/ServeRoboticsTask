import os
import tensorflow as tf
from dvc import api

'''This code downloads the MNIST dataset from a DVC (Data Version Control) repository, loads the data, and preprocesses the images. 
Using DVC ensures the dataset is versioned and the training process is reproducible.'''

# Download the MNIST dataset from DVC
os.makedirs('data', exist_ok=True)
api.get('data/mnist.pkl.gz', 'data/mnist.pkl.gz')

# Load and preprocess the MNIST dataset
with tf.io.gfile.GFile('data/mnist.pkl.gz', 'rb') as f:
    (x_train, y_train), (x_test, y_test), _ = tf.keras.datasets.mnist.load_data(f)

x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
