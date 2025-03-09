# for model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import math

# for GIFs
import glob
import imageio
import os
import sys
import PIL
import time

from IPython import display


# --------------------------------------- hyperparams ------------------------------------------------

BUFFER_SIZE = 60000 # unused for now, since i will set a seed to make debugging and optimizing easier
BATCH_SIZE = 128
NOISE_DIM = 100
FEATURE_MAP_SIZE = 7
CHANNELS = 256
SPATIAL_TENSOR = (FEATURE_MAP_SIZE, FEATURE_MAP_SIZE, CHANNELS)
FILTERS = (CHANNELS, 64, 1)
IMGS_TO_GEN = 4
EPOCHS = 10
tf.random.set_seed(420)

# --------------------------------------- data -------------------------------------------------------

# load the data from tensorflow
(train_imgs, train_lbls), (test_imgs, test_lbls) = tf.keras.datasets.mnist.load_data()

# reshape img tensor into (n_instances, px width, px height, channels)
# channels are the number of color channels in the image. this could be 1 for grayscale or 3 for RGB
# since our images are grayscale, set channels to 1
# transform them to float32, since we normalize the data to fall between 0 and 1
train_imgs = train_imgs.reshape(train_imgs.shape[0], 28, 28, 1)
train_imgs = train_imgs.astype("float32") / 255.0

test_imgs = test_imgs.reshape(test_imgs.shape[0], 28, 28, 1)
test_imgs = test_imgs.astype("float32") / 255.0

# create a tf dataset and make batches
# smaller batches hinder training potential, but are less memory intensive
# from docs:
# * for perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required
# * reshuffle_each_iteration controls whether the shuffle order should be different for each epoch (try without)
# * To shuffle an entire dataset, set buffer_size=dataset.cardinality(). This is equivalent to setting the
#   buffer_size equal to the number of elements in the dataset, resulting in uniform shuffle.
train_df = tf.data.Dataset.from_tensor_slices(train_imgs)
# train_df = train_df.batch(BATCH_SIZE).shuffle(BUFFER_SIZE)

# --------------------------------------- models ------------------------------------------------

def build_generator():
    """
    Start with a Dense layer that takes an input of 100 random values (noise) and transforms it into a
    vector of size 7*7*256=12544.

    BatchNormalization helps stabilize training by normalizing the activations of the layer before it.

    For hidden layers, LeakyRelu is better than relu, since it avoids "dying ReLU" problem and provides better
    gradient flow. Output layer should have tanh activation function, since it our values are normalized to be
    between -1 and 1, while sigmoid outputs values between 0 and 1.

    Reshape transforms 7*7*1=49-element vector into a 7x7x1 tensor, that can be interpreted as a 7x7px image
    with 1 channel (one feature map). It converts the vector representation into a spatial representation that
    can then be processed by convolutional layers.

    Core of our GAN network: Conv2DTranspose does upsampling (it increases the spatial dimensions). It inserts
    values into the input and performs a convolution. It adds padding and then applies a standard convolution
    operation to this expanded output. The kernel weigths are learned during training.
    """

    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(100,)))

    model.add(layers.Dense(math.prod(SPATIAL_TENSOR), use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape(SPATIAL_TENSOR)

    model.add(layers.Conv2DTranspose(FILTERS[0], (5, 5), strides=(1, 1), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(FILTERS[1],  (5, 5), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(FILTERS[2],   (5, 5), strides=(2, 2), padding="same", activation="tanh", use_bias=False))

    return model


"""
poglej si kako je discriminator izgrajen
"""
def build_discriminator():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# --------------------------------------- training ------------------------------------------------

generator = build_generator()
discriminator = build_discriminator()

# make an image
noise = tf.random.normal([1, 100])
gen_img = generator(noise, training=False)
plt.imshow(gen_img[0, :, :, 0])
plt.show()

# classify an image
decision = discriminator(gen_img)
print(decision)
