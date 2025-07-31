from utils import LR_SHAPE, HR_SHAPE
from typing import Generator
import tensorflow as tf
import keras
import numpy as np
import h5py
import os


BATCH_SIZE = 8
BUFFER_SIZE = 1000
BLOCK_LENGTH = 1


def h5_generator(filepath) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generator, ki vrača (lr, hr) pare slik iz H5 datotek z uporabo NumPy. Dimenzije HR slike so najprej
    prilagojene iz (320, 320) na (320, 320, 3). LR slike so pridobljene z bicubic downsampling HR slik.
    """
    with h5py.File(filepath.decode('utf-8'), "r") as hf:
        hr_images_raw = np.array(hf["reconstruction_rss"])

        channels = hr_images_raw.shape[0]

        for i in range(channels):
            hr_image = hr_images_raw[i].astype(np.float32)

            # zagotovi da je slika vsaj 3D (doda chanels dimenzijo, če manjka)
            if hr_image.ndim == 2:
                hr_image = np.expand_dims(hr_image, axis=-1) # (320, 320) -> (320, 320, 1)

            # če ima slika 2 kanala, vzamemo samo prvega (morda lahko ostranim)
            if hr_image.shape[-1] == 2:
                hr_image = hr_image[..., 0:1] # (320, 320, 2) -> (320, 320, 1)

            # zagotovi da ima 3 kanale (nisem prepričan ali je to pravi pristop)
            # np.tile ponovi array po določeni osi.
            if hr_image.shape[-1] == 1:
                hr_image = np.tile(hr_image, (1, 1, 3)) # (320, 320, 1) -> (320, 320, 3)

            assert hr_image.shape == HR_SHAPE

            # ustvari LR sliko z downsampling-om
            hr_image_tensor = tf.constant(hr_image)
            lr_image = tf.image.resize(hr_image_tensor, LR_SHAPE[:2], method=tf.image.ResizeMethod.BICUBIC)

            yield lr_image.numpy(), hr_image


def create_paired_dataset(filepath: list[str] | str) -> tf.data.Dataset:
    """
    Ustvari `tf.data.Dataset` s parnimi podatki LR in HR slik.
    """
    # `from_generator` spodaj pričakuje signiature izhodnih podatkov
    # vsak `yield` bo vrnil dve sliki, LR in HR, primernih oblik
    out_sig = (
        tf.TensorSpec(shape=LR_SHAPE, dtype=tf.float32),
        tf.TensorSpec(shape=HR_SHAPE, dtype=tf.float32)
    )

    # naredi dataset z imeni vseh datotek
    if isinstance(filepath, str):
        print("Creating paired dataset from", filepath)
        filepaths_pattern = os.path.join(filepath, "*.h5")
        filepaths_dataset = tf.data.Dataset.list_files(filepaths_pattern, shuffle=False)
    else:
        print("Creating paired dataset from", filepath[0].rsplit("/")[4])
        filepaths_dataset = tf.data.Dataset.from_tensor_slices(filepath)

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
    paired_dataset = paired_dataset.prefetch(tf.data.AUTOTUNE)  # za hitrejše nalaganje

    return paired_dataset