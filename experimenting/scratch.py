import os

ds_path = "../input/fastmri-knee/fastmri"
TRAIN_PATH = os.path.join(ds_path, "singlecoil_train")
TEST_PATH = os.path.join(ds_path, "singlecoil_test")
VAL_PATH = os.path.join(ds_path, "singlecoil_val")

def files_subset(path):
    MAX_FILES_TO_USE = 300
    all_files = glob.glob(os.path.join(path, "*.h5"))
    random.shuffle(all_files)
    return all_files[:MAX_FILES_TO_USE]

all_train_files = glob.glob(os.path.join(TRAIN_PATH, "*.h5"))
random.shuffle(all_train_files)
train_files = all_train_files[:MAX_FILES_TO_USE]

all_val_files = glob.glob(os.path.join(VAL_PATH, "*.h5"))
random.shuffle(all_val_files)
val_files = all_val_files[:int(MAX_FILES_TO_USE * 0.2)] # Npr. 20% validacijskih datotek

# Zdaj ustvarite nov dataset samo iz tega zmanjšanega seznama datotek
train = create_paired_dataset_from_list(train_files) # Potrebovali bomo novo funkcijo
val = create_paired_dataset_from_list(val_files)