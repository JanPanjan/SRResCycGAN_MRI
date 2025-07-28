from .src.utils import adam_opt
from .src.SRResCycGAN import SRResCycGAN
from .src.losses import PerceptualLoss, total_variation
from keras.losses import BinaryCrossentropy, MeanAbsoluteError

# TODO:
dataset = ...

model = SRResCycGAN(lambda_cyc=10.0, lambda_id=5.0)

losses = {
    "perceptual": PerceptualLoss(),
    "adversarial": BinaryCrossentropy(from_logits=True),
    "total_variation": total_variation,
    "content": MeanAbsoluteError(),
    "cyclic": MeanAbsoluteError(),
}


g_hr_optimizer = adam_opt()
d_hr_optimizer = adam_opt()
g_lr_optimizer = adam_opt()
d_lr_optimizer = adam_opt()

model.compile(
    g_hr_optimizer=g_hr_optimizer,
    d_hr_optimizer=d_hr_optimizer,
    g_lr_optimizer=g_lr_optimizer,
    d_lr_optimizer=d_lr_optimizer,
    losses=losses
)

model.fit(dataset, epochs=10)