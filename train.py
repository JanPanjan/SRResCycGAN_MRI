from src.utils import adam_opt
from src.data_loader import create_paired_dataset
from src.SRResCycGAN import SRResCycGAN
from src.losses import losses_dict

EPOCHS = 10
# TRAIN_PATH = "data/fastmri/singlecoil_train"
TRAIN_PATH = "experimenting"
VAL_PATH =  "data/fastmri/singlecoil_val"


if __name__ == "main":
    train = create_paired_dataset(TRAIN_PATH)
    val = create_paired_dataset(VAL_PATH)

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

    # ---- train -------------------------------------------------

    model.fit(train, epochs=EPOCHS)