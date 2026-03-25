import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow's version used in notebook:{}".format(tf.__version__))
print("TensorFlow datasets' version used in notebook{}".format(tfds.__version__))

(train_data, val_data, test_data), info = tfds.load('cifar10',
                                                    split=['train', 'test[:50%]', 'test[50%:]'],
                                                    as_supervised=True,
                                                    shuffle_files=False,
                                                    with_info=True)

print("The Number images in training set: {}".format(len(train_data)))
print("The number images in validation set: {}".format(len(val_data)))
print("The number images in test set: {}".format(len(test_data)))

info.splits['train'].num_examples
info.splits['test'].num_examples

info.features['label'].names
info.features['label'].num_classes

#Checking images
fig=tfds.show_examples(train_data, info)

#Preparing data
#Normalizing images
def preprocess(image, label):
    normalized_img=tf.cast(image, tf.float32)/255.0
    return normalized_img, label

def train_data_prep(data, shuffle_size, batch_size):
    
    data = data.map(preprocess)
    data = data.cache()
    data = data.shuffle(shuffle_size).repeat()
    data = data.batch(batch_size)
    data = data.prefetch(1)
    
    return data

def test_data_prep(data, batch_size):
    data = data.map(preprocess)
    data = data.batch(batch_size)
    data = data.cache()
    data = data.prefetch(1)
    
    return data

train_data_prepared = train_data_prep(train_data, 1000, 32)

test_data_prepared = test_data_prep(test_data, 32)

val_data_prepared = test_data_prep(val_data, 32)

#Model
input_shape = [32,32,3]


cifar_cnn = tf.keras.models.Sequential([
    # First Convolutional layer: 64 filters, kernel/filter size of 3

    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=input_shape),
  
    # First Pooling layer
    tf.keras.layers.MaxPooling2D(pool_size=2),

    # Second Convolutional layer & Pooling

    tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    # Third Convolutional & Pooling layer

    tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    # Flattening layer: For converting previous output into 1D column vector

    tf.keras.layers.Flatten(),

    # Fully Connected layers

    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),

    # Last layer: 10 neurons for 10 classes, activated by softmax
    tf.keras.layers.Dense(units=10, activation='softmax')

])

cifar_cnn.summary()