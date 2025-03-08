import matplotlib.pyplot as plt
import tensorflow as tf

(x, _), (_, _) = tf.keras.datasets.mnist.load_data()
fig, axes = plt.subplots(5, 5)

for i, ax in enumerate(axes.flatten()):
	ax.imshow(x[i])
	ax.axis("off")

plt.tight_layout()
plt.show()
