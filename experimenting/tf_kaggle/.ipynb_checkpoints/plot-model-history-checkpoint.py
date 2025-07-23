import pandas as pd
import matplotlib.pyplot as plt

his_df = pd.read_csv("model-history.csv", index_col=0)

plt.plot(his_df["loss"])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Validation Loss")
plt.grid(True)

plt.show()
