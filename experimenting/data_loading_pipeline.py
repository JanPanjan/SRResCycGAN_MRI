import tensorflow as tf
import keras
import numpy as np
import h5py
import os

BATCH_SIZE = 16
BUFFER_SIZE = 1000
BLOCK_LENGTH = 100


# filepath = os.path.join(".", "train-file.h5")
filepath = "../data/fastmri/singlecoil_train"
dims = set()

for f in os.listdir(filepath):
    with h5py.File(os.path.join(filepath, f), "r") as hf:
        d = np.array(hf["reconstruction_rss"])
        dims.add(d.shape)

dims

lr_image = lr_images[0]
lr_image.shape
x = tf.signal.ifft2d(lr_image)
x = tf.raw_ops.ComplexAbs(x=x)
x
x = keras.layers.CenterCrop(height=320, width=320, data_format="channels_last")(x)

x

x = tf.constant(np.expand_dims(lr_image, axis=-1))  # doda channels dimenzijo: (H, W) => (H, W, 1)
x
