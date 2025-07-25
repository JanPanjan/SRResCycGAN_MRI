import numpy as np
import fastmri
from fastmri.data import SliceDataset, transforms  # for FastMRI data utilities
import h5py  # for h5 data
import os
import matplotlib.pyplot as plt

root_dir = "/home/pogacha/progAAAAAAA/faks/GAN-osuprProject"
train_path = os.path.join(root_dir, "data/fastmri/singlecoil_train")
sample_path = os.path.join(train_path, "file1000001.h5")

# single h5 file that acts as a python dict
sample = h5py.File(sample_path)

dict(sample.attrs)
list(sample.keys())

slice = 10

# kspace image
kspace: h5py.Dataset = sample.get("kspace")
kspace.shape
kspace.dtype

sample_slice_kspace = kspace[slice]
sample_slice_kspace = np.log(np.abs(sample_slice_kspace) + 1e-9)  # complex -> float
plt.title(f"k-space slice {slice}")
plt.imshow(sample_slice_kspace)

# reconstructed image
rss: h5py.Dataset = sample.get("reconstruction_rss")
rss.shape
rss.dtype

sample_slice_rss = rss[slice]
plt.title(f"Reconstructed slice {slice}")
plt.imshow(sample_slice_rss, cmap="gray")

esc: h5py.Dataset= sample.get("reconstruction_esc")
esc.shape
esc.dtype

sample_slice_esc = esc[slice]
plt.imshow(sample_slice_esc, cmap="gray")
