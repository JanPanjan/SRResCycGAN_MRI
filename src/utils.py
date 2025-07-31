import glob
import os
import random
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers.schedules import ExponentialDecay
from keras.optimizers import Adam

LR_SHAPE = (80, 80, 3)
HR_SHAPE = (320, 320, 3)

def adam_opt(lr=1e-4, b1=0.9, b2=0.999, decay_steps=1e4, decay_rate=0.5):
    """
    Custom Adam optimizer z decaying learning rate.
    """
    lr_schedule = ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True  # da se lr spremeni na vsakih 10k korakov
    )
    return Adam(learning_rate=lr_schedule, beta_1=b1, beta_2=b2, weight_decay=False)


def scale_image(image):
    """
    Normalizira vrednosti od `image` na interval [0, 255].
    """
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val == min_val:  # prepreči ZeroDivision
        return image
    scaled = 255.0 * (image - min_val) / (max_val - min_val)
    return scaled.astype(np.uint8)


def files_subset(path) -> list[str]:
    """
    Vrne naključni manjši nabor datotek, ki se nahajajo na `path`.
    """
    MAX_FILES_TO_USE = 300
    all_files = glob.glob(os.path.join(path, "*.h5"))
    random.shuffle(all_files)
    return all_files[:MAX_FILES_TO_USE]


def hr_to_lr(lr_image):
    """
    HR sliko downsampla iz 320x320 v 80x80.
    """
    lr_image = np.expand_dims(lr_image, axis=-1)  # (320, 320, 1)
    lr_image = np.expand_dims(lr_image, axis=0)  # (1, 320, 320, 1)
    lr_image = np.tile(lr_image, (1, 1, 3))  # (1, 320, 320, 3)
    lr_image = tf.image.resize(tf.constant(lr_image), LR_SHAPE[:2], method=tf.image.ResizeMethod.BICUBIC)  # (1, 80, 80, 3)
    return lr_image


def calculate_slices(data_dir: str) -> int:
    """
    Izračuna število vseh slik, ki se nahajajo v datotekah v `data_dir`. Potrebno za
    izračun `steps_per_epoch` in `validation_steps` za `model.fit()`.
    """
    print("Calculating slices for ", data_dir, "...")
    total_slices = 0

    for f in os.listdir(data_dir):
        try:
            with h5py.File(os.path.join(data_dir, f), "r") as hf:
                total_slices += np.array(hf["reconstruction_rss"]).shape[0]
        except Exception as e:
            print(f"Exception occured: can't read file. {e}. Skipping.")
    print("Total slices:", total_slices)

    return total_slices


def plot_history(history_obj: dict, epochs: int):
    """
    Prikaže graf losses v odvisnosti od epoch.
    """
    for k in history_obj.keys():
        plt.plot(epochs, history_obj[k], label=k)

    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()