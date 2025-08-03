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


# from generator
# %%
import tensorflow as tf

# %%
def count(stop):
  i = 0
  while i<stop:
    yield i, i+1
    i += 1

for batch in count(5):
    print(batch)
# %%
counter = tf.data.Dataset.from_generator(generator=count, args=[25], output_types=tf.int32, output_shapes = (2,))

for batch in counter:
    print(batch)

# %%
import h5py
import glob
import numpy as np
import os

file_path = "../data/fastmri/train-sub"

for f in glob.glob(''.join([file_path, "/*"])):
    print(f)

# %%
file_paths = tf.io.gfile.glob(os.path.join(file_path, '*.h5'))
print(file_paths)

# %%
# fnames_ds = tf.data.Dataset.list_files(glob.glob(''.join([file_path, "/*"])))
fnames_ds = tf.data.Dataset.from_tensor_slices(file_paths)

# %%
for name in fnames_ds:
    print(name)

# %%
def process_h5(file_path):
    """
    Odpre h5 datoteko
    """
    path = file_path.numpy().decode('utf-8')
    with h5py.File(path, "r") as f:
        hr_mono = np.array(f["reconstruction_rss"])
        hr_rgb = np.repeat(hr_mono[..., np.newaxis], 3, axis=-1)  # (B, 320, 320, 3)

        hr_4d_mono = tf.constant(hr_mono)[..., tf.newaxis]  # (B, 320, 320, 1)
        lr_4d_mono = tf.image.resize(hr_4d_mono, [80, 80], method=tf.image.ResizeMethod.BICUBIC)  # (B, 80, 80, 1)
        lr_4d_rgb = tf.tile(lr_4d_mono, [1, 1, 1, 3])  # (B, 80, 80, 3)
        lr_rgb = lr_4d_rgb.numpy()

        # (B, 320, 320, 3), (B, 80, 80, 3)
        return hr_rgb.astype(np.float32), lr_rgb.astype(np.float32)

# %%
@tf.function
def map_func(path_tensor):
    rss1, rss2 = tf.py_function(
        process_h5,
        [path_tensor],
        (tf.float32, tf.float32)
    )
    rss1.set_shape([None, 320, 320, 3])
    rss2.set_shape([None, 80, 80, 3])
    tf.print(rss1.shape, rss2.shape)
    return tf.data.Dataset.from_tensor_slices((rss1, rss2))

# %%
BATCH_SIZE = 16
rss_ds = fnames_ds.flat_map(map_func).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

for h, l in rss_ds:
    pass

# %%
# -------------------------------------------------------------------------------------------------
# --- OPTIMIZIRAN PRISTOP 1 ---

# 1. Funkcija za py_function samo naloži surove podatke
def nalozi_surovi_blok(file_path_tensor):
    path = file_path_tensor.numpy().decode('utf-8')
    with h5py.File(path, "r") as f:
        # Vrnemo samo enokanalne, neobdelane podatke
        return np.array(f["reconstruction_rss"]).astype(np.float32)

# %%
@tf.function
def ustvari_dataset_surovih_rezin(file_path_tensor):
    """Vrne dataset enokanalnih rezin iz ene datoteke."""
    surovi_blok = tf.py_function(
        nalozi_surovi_blok,
        [file_path_tensor],
        tf.float32
    )
    surovi_blok.set_shape([None, 320, 320])
    return tf.data.Dataset.from_tensor_slices(surovi_blok)

# %%
@tf.function
def ustvari_hr_lr_par(mono_slika):
    """
    Iz ene enokanalne slike ustvari HR in LR par s 3 kanali.
    Ta funkcija uporablja samo TensorFlow operacije in je zato zelo hitra.
    """
    # Dodamo os za kanal: (320, 320) -> (320, 320, 1)
    mono_slika_4d = mono_slika[..., tf.newaxis]

    # Ustvarimo HR sliko (3 kanali)
    hr_slika = tf.tile(mono_slika_4d, [1, 1, 3]) # (320, 320, 3)

    # Ustvarimo LR sliko (3 kanali)
    lr_mono = tf.image.resize(mono_slika_4d, [80, 80])
    lr_slika = tf.tile(lr_mono, [1, 1, 3]) # (80, 80, 3)

    return lr_slika, hr_slika

# %%
# Zgradimo končni cevovod
data_dir = "../data/fastmri/train-sub"
pattern = os.path.join(data_dir, '*.h5')
file_paths = tf.io.gfile.glob(pattern)
file_paths

# %%
files_ds = tf.data.Dataset.from_tensor_slices(file_paths)
[ print(f) for f in files_ds ]

# %%
# flat_map ustvari tok enokanalnih slik
surove_rezine_ds = files_ds.flat_map(ustvari_dataset_surovih_rezin)
[ print(f.shape) for f in surove_rezine_ds.take(5) ]

# %%
# .map nato vsako sliko hitro pretvori v (LR, HR) par
optimiziran_ds_1 = surove_rezine_ds.map(
    ustvari_hr_lr_par,
    num_parallel_calls=tf.data.AUTOTUNE
)

# %%
# Dodamo še standardne optimizacije
optimiziran_ds_1 = optimiziran_ds_1.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
[ print(h.shape, l.shape) for h,l in optimiziran_ds_1.take(1) ]

# %%
# --- OPTIMIZIRAN PRISTOP 2 ---

# 1. Funkcija za py_function, ki naloži in vektorizirano obdela celoten blok
def nalozi_in_obdelaj_blok_vektorsko(file_path_tensor):
    """Naloži cel blok in ga v celoti pretvori v HR in LR RGB bloke."""
    path = file_path_tensor.numpy().decode('utf-8')
    with h5py.File(path, "r") as f:
        hr_block_mono = np.array(f["reconstruction_rss"]).astype(np.float32)

    # Vektorizirana obdelava (enaka kot v prejšnjem odgovoru)
    hr_block_rgb = np.repeat(hr_block_mono[..., np.newaxis], 3, axis=-1)

    hr_tensor_4d_mono = tf.constant(hr_block_mono)[..., tf.newaxis]
    lr_tensor_4d_mono = tf.image.resize(hr_tensor_4d_mono, [80, 80])
    lr_tensor_4d_rgb = tf.tile(lr_tensor_4d_mono, [1, 1, 3])
    lr_block_rgb = lr_tensor_4d_rgb.numpy()

    return hr_block_rgb, lr_block_rgb

@tf.function
def ustvari_dataset_iz_datoteke_opt(filepath):
    """Funkcija, ki jo kliče interleave. Vrne dataset parov iz ene datoteke."""
    hr_block, lr_block = tf.py_function(
        nalozi_in_obdelaj_blok_vektorsko,
        [filepath],
        (tf.float32, tf.float32)
    )
    hr_block.set_shape([None, 320, 320, 3])
    lr_block.set_shape([None, 80, 80, 3])

    return tf.data.Dataset.from_tensor_slices((lr_block, hr_block)) # Opomba: obrnil sem par, da je (LR, HR)

# %%
# Zgradimo končni cevovod
data_dir = "../data/fastmri/train-sub"
file_paths = os.path.join(data_dir, '*.h5')
file_paths

# %%
filepaths_dataset = tf.data.Dataset.list_files(, shuffle=False)

# %%
# interleave kliče funkcijo, ki vektorsko obdela vsako datoteko
optimiziran_ds_2 = filepaths_dataset.interleave(
    ustvari_dataset_iz_datoteke_opt,
    cycle_length=tf.data.AUTOTUNE,
    num_parallel_calls=tf.data.AUTOTUNE
)

# %%
# Dodamo še standardne optimizacije
optimiziran_ds_2 = optimiziran_ds_2.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)