"""
just use tensorflow
"""
import numpy as np
import os
from rich.console import Console

console = Console()

# podatke išče v 'data' directoriju. če zaženeš script iz 'src/', potem bo tu error
# raje zaženi iz root: 'src/data-wrangling.py'
if "data" not in os.listdir():
    raise ValueError("Error: directory 'data' not found in current direcory")

PATH = "data/mnist.npz"

console.log(f"Reading data from '{PATH}'")
with np.load(PATH) as data:
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]

# kako izgledajo podatki?
print("shape of data:")
for set in [x_train, x_test, y_train, y_test]:
    print(f"{set.shape}")

# 70'000 slik
# vsaka slika je 28x28 matrika
# kakšne vrednosti imajo x in y datasets?
for set in [x_train, x_test]:
    print(f"min: {min(np.unique(set))}, max: {max(np.unique(set))}, count: {len(np.unique(set))}")

# 256 različnih vrednosti med 0 in 255
# moram jih zmanjšat na interval [0,1]
# podatki slik so 2D arrays z vrednostmi torej 0, 1, 2,..., 254, 255
# najlažje je najprej flatten-at podatke
console.log("Flattening data")
x_train = x_train.flatten()
x_test = x_test.flatten()

# vsako vrednost je treba delit z največjo vrednostjo

def normalize(np_set: np.ndarray, factor=None) -> np.ndarray:
    """
    Normalized the given numpy array by its max value, unless 'factor' is given

    Args:
        np_set (np.ndarray): numpy array to be normalized
        factor (None, np.number): factor with which to normalize np_set values
    Returns:
        np.ndarray: normalized numpy array
    """
    if factor:
        assert isinstance(factor, np.number), "Error: 'factor' must be a of type 'np.number'"
        return np_set / factor
    else:
        max_arg = max(np_set)
        return np_set / max_arg

console.log("Normalizing sets")
norm_x_test = normalize(x_test)
norm_x_train = normalize(x_test)

req_min = 0.0
req_max = 1.0

# zdaj bi morala biti min vrednost 0 in max 1
assert min(norm_x_test.tolist()) == req_min, f"Error: min value in 'norm_x_test' is {min(norm_x_test.tolist())} and is not {req_min}"
assert max(norm_x_test.tolist()) == req_max, f"Error: max value in 'norm_x_test' is {max(norm_x_test.tolist())} and is not {req_max}"
assert min(norm_x_train.tolist()) == req_min, f"Error: min value in 'norm_x_train' is {min(norm_x_train.tolist())} and is not {req_min}"
assert max(norm_x_train.tolist()) == req_max, f"Error: max value in 'norm_x_train' is {max(norm_x_train.tolist())} and is not {req_max}"
