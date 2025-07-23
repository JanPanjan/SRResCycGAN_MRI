import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, Conv2DTranspose, LayerNormalization, Dropout, LeakyReLU, ReLU

def encoder(filters, kernel_size, do_norm=False):
    """
    filters = input size
    if training, set do_norm to True
    Conv -> Instance Norm -> Leaky ReLU
    """
    m = Sequential()
    m.add(Conv2D(
        filters,
        kernel_size,
        strides=2,
        padding="same",
        kernel_initializer="random_normal", # NOTE: will this kernel-init. work?
        use_bias=False
    ))
    if do_norm:
        m.add(LayerNormalization())
    m.add(LeakyReLU())
    return m


def decoder(filters, kernel_size, do_dropout=False):
    """
    if training, set do_dropuout to True
    Transposed Conv -> Instance Norm -> Dropout -> ReLU
    """
    m = Sequential()
    m.add(Conv2DTranspose(
        filters,
        kernel_size,
        strides=2,
        padding="same",
        kernel_initializer="random_normal", # NOTE: will this kernel-init. work?
        use_bias=False
    ))
    if do_dropout:
        m.add(Dropout(0.5))
    m.add(ReLU())
    return m


def Generator():
    """
    Encoder -> Resnet -> Decoder
    """
    pass


def build_discriminator():
    pass
