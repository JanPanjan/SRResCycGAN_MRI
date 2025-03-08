"""
Kaggle TensorFlow Course
https://www.kaggle.com/code/janpanjan/exercise-stochastic-gradient-descent/edit
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split

fuel = pd.read_csv('data/fuel.csv')
X = fuel.copy()
y = X.pop('FE')
preprocessor = make_column_transformer(
    (StandardScaler(),
     make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse_output=False),
     make_column_selector(dtype_include=object)),
)
X = preprocessor.fit_transform(X)
y = np.log(y) # log transform target instead of standardizing
input_shape = [X.shape[1]]
print(f"Input shape: {input_shape}\n")

print(f"original data:\n {fuel.head()}\n")
print(f"processed features:\n {pd.DataFrame(X[:10,:]).head()}\n")

# ---------------------------------------------------------------------------------

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1),
])
model.compile(
    optimizer="adam",
    loss="mae",
)
his = model.fit(
    X, y,
    batch_size=128,
    epochs=100
)

# ---------------------------------------------------------------------------------

import pandas as pd

history_df = pd.DataFrame(his.history)
try:
    history_df.to_csv('/home/pogacha/progAAAAAAA/GAN-osuprProject/src/tf_kaggle/model-history.csv')
    print("Model history exported to 'src/tf_kaggle/model-history.csv'")
except Exception as e:
    print(f"Error: {e}")
