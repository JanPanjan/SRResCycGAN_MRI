from .utils import LR_SHAPE, HR_SHAPE
from typing import Generator
import tensorflow as tf
import keras
import numpy as np
import h5py
import os

BATCH_SIZE = 16
BUFFER_SIZE = 1000
BLOCK_LENGTH = 1


def kspace_transform(kspace_data) -> keras.KerasTensor:
    """
    k-space podatke transformira v LR sliko dimenzije 320x320
    """
    x = tf.signal.ifft2d(kspace_data)
    x = tf.raw_ops.ComplexAbs(x=x)
    x = keras.layers.CenterCrop(height=320, width=320, data_format="channels_last")(x)
    return x


def h5_generator(filepath) -> Generator[tuple[tf.Tensor, tf.Tensor], None, None]:
    """
    Generator, ki vrača (lr, hr) pare slik iz H5 datotek.
    """
    with h5py.File(filepath, "r") as hf:
        hr_images = np.array(hf["reconstruction_rss"])
        lr_images = np.array(hf["kspace"])

        # poskrbi da so parni podatki, tudi če nimata kspace in rss enakega števila sliceov
        channels = hr_images.shape[0] if hr_images.shape[0] > lr_images.shape[0] else lr_images.shape[0]

        for i in range(channels):
            # vsaki sliki doda channels dimenzijo: (H, W) => (H, W, 1)
            lr_image = tf.constant(np.expand_dims(lr_images[i], axis=-1))
            hr_image = tf.constant(np.expand_dims(hr_images[i], axis=-1))
            assert lr_image.shape[2] == 1
            assert hr_image.shape[2] == 1

            # transformira kspace podatke v sliko
            lr_transformed = kspace_transform(lr_image)
            # opravi bicubic downsampling, kot omenjajo v članku
            lr_transformed = tf.image.resize(images=lr_transformed, size=(80, 80), method=tf.image.ResizeMethod.BICUBIC)
            assert lr_transformed.shape == (80, 80, 1)

            # poskrbi da ima tudi HR pravo dimenzijo
            if hr_image.shape != (320, 320, 1):
                hr_image = tf.image.resize(images=hr_image, size=(320, 320), method=tf.image.ResizeMethod.BICUBIC)

            assert hr_image.shape == (320, 320, 1)

            # pretvori v rgb s 3 kanali: (H, W, 1) => (H, W, 3)
            lr_image = tf.image.grayscale_to_rgb(lr_transformed)
            hr_image = tf.image.grayscale_to_rgb(hr_image)
            assert lr_image.shape == (80, 80, 3)
            assert hr_image.shape == (320, 320, 3)

            yield lr_image, hr_image


def create_paired_dataset(data_dir) -> tf.data.Dataset:
    """
    Ustvari `tf.data.Dataset` s parnimi podatki LR in HR slik. LR slike predstavljajo transformirani k-space
    podatki, HR pa rekonstruirane in prečiščene slike dostopne znotraj datotek pod `reconstruction_rss`.
    """
    filepaths_pattern = os.path.join(data_dir, "*.h5")

    # `from_generator` spodaj pričakuje signiature izhodnih podatkov
    # vsak `yield` bo vrnil dve sliki, LR in HR, primernih oblik
    out_sig = (
        tf.TensorSpec(shape=tf.TensorShape(LR_SHAPE), dtype=tf.float32),
        tf.TensorSpec(shape=tf.TensorShape(HR_SHAPE), dtype=tf.float32)
    )

    # dataset z imeni vseh datotek
    filepaths_dataset = tf.data.Dataset.list_files(filepaths_pattern, shuffle=False)

    # z generatorjem odpre vsako datoteko posebej in postopoma z `yield` vrača pare slik
    paired_dataset = filepaths_dataset.interleave(
        lambda filepath: tf.data.Dataset.from_generator(
            generator=h5_generator,
            output_signature=out_sig,
            args=(filepath,)  # vejica, ker mora biti sequence
        ),
        cycle_length=tf.data.AUTOTUNE,       # število vhodnih elementov, ki se procesirajo istočasno
        block_length=BLOCK_LENGTH,           # da zajame vse slike v datoteki, ker ni fiksnega števila slojev
        num_parallel_calls=tf.data.AUTOTUNE  # za hitrejše procesiranje
    )

    paired_dataset = paired_dataset.shuffle(buffer_size=BUFFER_SIZE)
    paired_dataset = paired_dataset.batch(BATCH_SIZE)
    paired_dataset = paired_dataset.prefetch(tf.data.AUTOTUNE)

    return paired_dataset


if __name__ == "__main__":
    print("Testiram delovanje `create_paired_dataset`")
    data_dir = "experimenting"
    dataset = create_paired_dataset(data_dir)
    print(dataset.take(1).element_spec)
