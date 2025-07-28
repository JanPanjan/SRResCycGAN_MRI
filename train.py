from .src.utils import adam_opt
from .src.SRResCycGAN import SRResCycGAN
from .src.losses import losses_dict

# TODO:
dataset = ...

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

model.fit(dataset, epochs=10)