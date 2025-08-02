import tensorflow as tf
import os

# Assume your 'generate_paired_data' function has already run
# and populated these directories.
hr_path = "../data/mnist/train/hr_320x320"
lr_path = "../data/mnist/train/lr_80x80"

# --- Step 1: Get Sorted File Paths ---
lr_image_paths = sorted([os.path.join(lr_path, fname) for fname in os.listdir(lr_path)])
hr_image_paths = sorted([os.path.join(hr_path, fname) for fname in os.listdir(hr_path)])

print(f"Found {len(lr_image_paths)} low-resolution images.")
print(f"Found {len(hr_image_paths)} high-resolution images.")

# --- Step 2: Create a Loading and Processing Function ---
def load_and_process_pair(lr_path, hr_path):
    # Read files from disk
    lr_img_raw = tf.io.read_file(lr_path)
    hr_img_raw = tf.io.read_file(hr_path)

    # Decode PNGs into tensors (specify 1 channel for grayscale)
    lr_tensor = tf.io.decode_png(lr_img_raw, channels=1)
    hr_tensor = tf.io.decode_png(hr_img_raw, channels=1)

    # Convert to float and normalize to [0, 1] range
    lr_tensor = tf.image.convert_image_dtype(lr_tensor, tf.float32)
    hr_tensor = tf.image.convert_image_dtype(hr_tensor, tf.float32)

# --- Step 3: Build the tf.data Pipeline ---
# Create a dataset of file path pairs
path_dataset = tf.data.Dataset.from_tensor_slices((lr_image_paths, hr_image_paths))

# Use .map() to apply the loading function to each pair
# num_parallel_calls=tf.data.AUTOTUNE lets TF process multiple images in parallel
image_dataset = path_dataset.map(load_and_process_pair, num_parallel_calls=tf.data.AUTOTUNE)

# Configure the dataset for optimal training performance
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_dataset = image_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# The 'train_dataset' is now ready to be passed to model.fit()
print("\ntf.data.Dataset created successfully.")


# --- Example: Inspect a single batch ---
lr_batch, hr_batch = next(iter(train_dataset))
print(f"LR batch shape: {lr_batch.shape}")
print(f"HR batch shape: {hr_batch.shape}")