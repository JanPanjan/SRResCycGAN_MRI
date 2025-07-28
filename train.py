from .src.utils import adam_opt
from .src.data_loader import create_paired_dataset
from .src.SRResCycGAN import SRResCycGAN
from .src.losses import losses_dict

TRAIN_PATH = ...
VAL_PATH = ...

# TODO:
train = create_paired_dataset(TRAIN_PATH)
val = create_paired_dataset(VAL_PATH)

"""
# Vizualizacija prvih nekaj slik (za debug)
for lr_batch, hr_batch in dataset.take(1):
    print(f"Oblika LR serije: {lr_batch.shape}") # (BATCH_SIZE, 1, 320, 320)
    print(f"Oblika HR serije: {hr_batch.shape}")
    # Pretvori v NumPy array
    lr_image = lr_batch[0, 0].numpy() # Prva slika, prvih kanalov (1, 320, 320)
    hr_image = hr_batch[0, 0].numpy()

    # Prikaz slik
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(lr_image, cmap='gray') # Če so sive slike
    plt.title('LR slika')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(hr_image, cmap='gray')
    plt.title('HR slika')
    plt.axis('off')
    plt.show()
    break # Prikaz samo prvega batcha (za hitrejši test)
"""

model = SRResCycGAN(lambda_cyc=10.0, lambda_content=5.0)

g_hr_optimizer = adam_opt()
d_hr_optimizer = adam_opt()
g_lr_optimizer = adam_opt()
d_lr_optimizer = adam_opt()

model.compile(
    g_hr_optimizer=g_hr_optimizer,
    d_hr_optimizer=d_hr_optimizer,
    g_lr_optimizer=g_lr_optimizer,
    d_lr_optimizer=d_lr_optimizer,
    losses=losses_dict
)

model.fit(train, epochs=10)