from typing import Generator
import tensorflow as tf
import keras
import numpy as np
import h5py
import os

BATCH_SIZE = 16
BUFFER_SIZE = 1000
BLOCK_LENGTH = 100


def kspace_transform(kspace_data) -> np.ndarray:
    """
    k-space podatke transformira v LR slike dimenzije 320x320
    """
    x = tf.signal.ifft2d(kspace_data)
    x = tf.raw_ops.ComplexAbs(x=x)
    x = keras.layers.CenterCrop(height=320, width=320, data_format="channels_first")(x)
    x = np.array(x)
    return x


def h5_generator(filepath) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generator, ki vrača (lr, hr) pare slik iz H5 datoteke.
    """
    with h5py.File(filepath, "r") as hf:
        hr_images = np.array(hf["reconstruction_rss"])
        lr_images = np.array(hf["kspace"])
        lr_transformed = kspace_transform(lr_images)

        assert hr_images.shape == lr_transformed.shape

        for i in range(lr_transformed.shape[0]):
            # doda channels dimenzijo obema 320x320 tensorjema => (1, 320, 320)
            lr_out = np.expand_dims(lr_transformed[i], axis=0)
            hr_out = np.expand_dims(hr_images[i], axis=0)
            yield lr_out, hr_out


def create_paired_dataset(data_dir) -> tf.data.Dataset:
    """
    Ustvari `tf.data.Dataset` s parnimi podatki LR in HR slik. LR slike predstavljajo
    transformirani k-space podatki, HR pa rekonstruirane in prečiščene slike dostopne znotraj
    H5 datoteke pod imenom `reconstruction_rss`.
    """
    filepaths_pattern = os.path.join(data_dir, "*.h5")

    # `from_generator` pričakuje signiature izhodnih podatkov
    # vsak `yield` bo vrnil dve sliki, LR in HR, obe obliki 1x320x320
    lr_shape = tf.TensorShape([1, 320, 320])
    hr_shape = tf.TensorShape([1, 320, 320])
    out_sig = (
        tf.TensorSpec(shape=lr_shape, dtype=tf.float32),
        tf.TensorSpec(shape=hr_shape, dtype=tf.float32)
    )

    # dataset z imeni vseh datotek
    filepaths_dataset = tf.data.Dataset.list_files(filepaths_pattern, shuffle=False)

    # s pomočjo generatorja odpre vsako datoteko posebej in postopoma z `yield` vrača pare slik
    paired_dataset = filepaths_dataset.interleave(
        lambda filepath: tf.data.Dataset.from_generator(
            generator=h5_generator,
            output_signature=out_sig,
            args=(filepath,)
        ),
        cycle_length=tf.data.AUTOTUNE,       # število vhodnih elementov, ki se procesirajo istočasno
        block_length=BLOCK_LENGTH,           # da zajame vse slike v datoteki, ker ni fiksnega števila slojev
        num_parallel_calls=tf.data.AUTOTUNE  # za hitrejše procesiranje
    )

    paired_dataset = paired_dataset.shuffle(buffer_size=BUFFER_SIZE)
    paired_dataset = paired_dataset.batch(BATCH_SIZE)
    paired_dataset = paired_dataset.prefetch(tf.data.AUTOTUNE)

    return paired_dataset


paired_dataset = create_paired_dataset(".")
print("\nTaking 2 batches from the dataset:")
for i, (lr_batch, hr_batch) in enumerate(paired_dataset.take(100)):
    print(f"Batch {i+1}: LR shape: {lr_batch.shape}, HR shape: {hr_batch.shape}")
