from keras import Model, Loss, Optimizer
from .models import Generator_HR, Generator_LR, Discriminator_HR, Discriminator_LR
from .utils import LR_SHAPE, HR_SHAPE


class SRResCycGAN(Model):
    def __init__(self, lambda_cyc=10.0, lambda_id=5.0) -> None:
        super(SRResCycGAN, self).__init__()

        self.G_HR = Generator_HR(input_shape=LR_SHAPE)
        self.G_LR = Generator_LR(input_shape=HR_SHAPE)
        self.D_HR = Discriminator_HR(input_shape=HR_SHAPE)
        self.D_LR = Discriminator_LR(input_shape=LR_SHAPE)
        self.lambda_cyc = lambda_cyc
        self.lambda_id = lambda_id


    def compile(self,
        g_hr_optimizer: Optimizer,
        d_hr_optimizer: Optimizer,
        g_lr_optimizer: Optimizer,
        d_lr_optimizer: Optimizer,
        losses: dict[str, Loss]
    ):
        super(SRResCycGAN, self).compile()

        self.g_hr_optimizer = g_hr_optimizer
        self.d_hr_optimizer = d_hr_optimizer
        self.g_lr_optimizer = g_lr_optimizer
        self.d_lr_optimizer = d_lr_optimizer

        self.adv_loss = losses["adversarial"]
        self.cyc_loss = losses["cyclic"]
        self.content_loss = losses["content"]
        self.perceptual_loss = losses["perceptual"]
        self.tv_loss = losses["total_variation"] # Gradient difference loss


    def train_step():
        pass