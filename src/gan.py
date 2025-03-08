# for model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# for GIFs
import glob
import imageio
import os
import sys
import PIL
import time

from IPython import display


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

# create a tf dataset, make batches and shuffle elements
# smaller batches hinder training potential, but are less memory intensive
# from docs:
# * for perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required
# * reshuffle_each_iteration controls whether the shuffle order should be different for each epoch (try without)
# * To shuffle an entire dataset, set buffer_size=dataset.cardinality(). This is equivalent to setting the
#   buffer_size equal to the number of elements in the dataset, resulting in uniform shuffle.
BATCH_SIZE = 128
BUFFER_SIZE = train_imgs.shape[0]
channels = 1

train_df = tf.data.Dataset.from_tensor_slices(train_imgs)
train_df = train_df.batch(BATCH_SIZE).shuffle(BUFFER_SIZE)

def build_generator():
    model = tf.keras.Sequential()

    # start with a random vector of size 100, that represents noise
    model.add(layers.Input(shape=(100,)))
    # x*y*z: x*y is img size while z is number of channels
    # more channels == more depth, but since we have grayscale images, I think
    # 1 channel will be enough for this model
    model.add(layers.Dense(units=7*7*channels))

    return model


def build_discriminator():
    model = tf.keras.Sequential()

    return model


generator = build_generator()
print(generator.summary())
discriminator = build_discriminator()
