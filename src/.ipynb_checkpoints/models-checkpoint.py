import tensorflow as tf
import tensorflow_addons as tfa
from keras import Layer, layers


"""
Residual block za generator. Uporablja naslednje layerje:
- Parametrized ReLU (PReLU)
- Convolution layer
- Elementwise sum
- Instance normalization
"""
class ResidualBlock(Layer):
    # norm_method: InstanceNormalization ali BatchNormalization Layer za ta projekt
    def __init__(self, norm_method: Layer, filters, kernel_size, strides, **kwargs):
        super().__init__(**kwargs)
        # self.conv = layers.Conv2D(filters, kernel_size, strides=strides, padding)
        self.conv = layers.Conv2D(filters, kernel_size, strides, padding="same")

        assert norm_method == layers.BatchNormalization || norm_method == layers.InstanceNorm
        self.norm = norm_method

    # ko se pokliče nad inputs se zgodi to kar je definirano tu notri
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.norm(inputs)


class Generator():
    pass


class Discriminator():
    pass