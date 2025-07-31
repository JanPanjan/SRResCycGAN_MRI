import tensorflow as tf
from models import Generator_HR, Generator_LR, Discriminator_HR, Discriminator_LR
from utils import LR_SHAPE, HR_SHAPE
from keras import Model, Loss, Optimizer
from tensorflow import GradientTape, ones_like, zeros_like


class SRResCycGAN(Model):
    """
    Super Resolution Residual Cyclic GAN
    """
    def __init__(self, lambda_cyc=10.0, lambda_content=5.0) -> None:
        super(SRResCycGAN, self).__init__()

        self.G_HR = Generator_HR(input_shape=LR_SHAPE)
        self.G_LR = Generator_LR(input_shape=HR_SHAPE)
        self.D_HR = Discriminator_HR(input_shape=HR_SHAPE)
        self.D_LR = Discriminator_LR(input_shape=LR_SHAPE)
        self.lambda_cyc = lambda_cyc
        self.lambda_content = lambda_content

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
        self.tv_loss = losses["total_variation"]

    @tf.function
    def train_step(self, data):
        """
        Vsak korak treniranja se generirajo HR in LR slike z generatorjema, ki se nato ocenita z diskriminatorjema.
        Diskriminatorja uporabljata kot loss standardni adversarial loss za GAN-se, implementiran z BinaryCrossEntropy.
        Generatorja uporabljata poleg adversarial loss še perceptual, total variation, content in cyclic loss (posebnost
        Cyclic-consistent modelov). Content in cyclic sta uteženi, s čimer omogoči modelu, da pripomoreta več k izgubi.
        """
        real_lr, real_hr = data

        with GradientTape(persistent=True) as disc_tape:
            # generira slike
            fake_hr = self.G_HR(real_lr, training=True)
            fake_lr = self.G_LR(real_hr, training=True)

            # diskriminira
            real_hr_pred = self.D_HR(real_hr, training=True)
            fake_hr_pred = self.D_HR(fake_hr, training=True)

            real_lr_pred = self.D_LR(real_lr, training=True)
            fake_lr_pred = self.D_LR(fake_lr, training=True)

            # discriminator losses
            dhr_real_loss = self.adv_loss(ones_like(real_hr_pred), real_hr_pred)
            dhr_fake_loss = self.adv_loss(zeros_like(fake_hr_pred), fake_hr_pred)
            dhr_total_loss = (dhr_real_loss + dhr_fake_loss) * 0.5

            dlr_real_loss = self.adv_loss(ones_like(real_lr_pred), real_lr_pred)
            dlr_fake_loss = self.adv_loss(zeros_like(fake_lr_pred), fake_lr_pred)
            dlr_total_loss = (dlr_real_loss + dlr_fake_loss) * 0.5

            total_disc_loss = dhr_total_loss + dlr_total_loss


        with GradientTape(persistent=True) as gen_tape:
            # generira slike
            fake_hr = self.G_HR(real_lr, training=True)
            fake_lr = self.G_LR(real_hr, training=True)
            cycled_hr = self.G_HR(fake_lr, training=True)
            cycled_lr = self.G_LR(fake_hr, training=True)

            # diskriminira za adversarial loss
            fake_hr_pred_gen = self.D_HR(fake_hr, training=True)
            fake_lr_pred_gen = self.D_LR(fake_lr, training=True)

            # generator losses
            perceptual_loss = self.perceptual_loss(real_hr, fake_hr)

            ghr_adv = self.adv_loss(ones_like(fake_hr_pred_gen), fake_hr_pred_gen)
            glr_adv = self.adv_loss(ones_like(fake_lr_pred_gen), fake_lr_pred_gen)
            adv_loss = ghr_adv + glr_adv

            tv_loss = self.tv_loss(real_hr, fake_hr)

            content_loss = self.content_loss(real_hr, fake_hr)

            # cyclic loss
            cyc_forward = self.cyc_loss(cycled_lr, real_lr)
            cyc_backward = self.cyc_loss(cycled_hr, real_hr)
            cyc_loss = cyc_forward + cyc_backward

            total_gen_loss = perceptual_loss + adv_loss + tv_loss + \
                            (content_loss * self.lambda_content) + \
                            (cyc_loss * self.lambda_cyc)


        # gradienti diskriminatorja
        dhr_grads = disc_tape.gradient(dhr_total_loss, self.D_HR.trainable_variables)
        dlr_grads = disc_tape.gradient(dlr_total_loss, self.D_LR.trainable_variables)

        self.d_hr_optimizer.apply_gradients(zip(dhr_grads, self.D_HR.trainable_variables))
        self.d_lr_optimizer.apply_gradients(zip(dlr_grads, self.D_LR.trainable_variables))

        # gradienti generatorja
        ghr_grads = gen_tape.gradient(total_gen_loss, self.G_HR.trainable_variables)
        glr_grads = gen_tape.gradient(total_gen_loss, self.G_LR.trainable_variables)

        self.g_hr_optimizer.apply_gradients(zip(ghr_grads, self.G_HR.trainable_variables))
        self.g_lr_optimizer.apply_gradients(zip(glr_grads, self.G_LR.trainable_variables))

        return {
            "Total discriminator loss": total_disc_loss,
            "Total generator loss": total_gen_loss,
            "Perceptual loss": perceptual_loss,
            "Adversarial loss": adv_loss,
            "Total variation loss": tv_loss,
            "Content loss": content_loss,
            "Cyclic loss": cyc_loss
        }

    @tf.function
    def test_step(self, data):
        """
        Podobno kot pri `train_step`.
        """
        real_lr, real_hr = data

        # training=False ker nočemo da posodablja uteži
        fake_hr = self.G_HR(real_lr, training=False)
        fake_lr = self.G_LR(real_hr, training=False)
        cycled_hr = self.G_HR(fake_lr, training=False)
        cycled_lr = self.G_LR(fake_hr, training=False)

        real_hr_pred = self.D_HR(real_hr, training=False)
        fake_hr_pred = self.D_HR(fake_hr, training=False)
        real_lr_pred = self.D_LR(real_lr, training=False)
        fake_lr_pred = self.D_LR(fake_lr, training=False)

        # enako kot train_step
        dhr_real_loss = self.adv_loss(ones_like(real_hr_pred), real_hr_pred)
        dhr_fake_loss = self.adv_loss(zeros_like(fake_hr_pred), fake_hr_pred)
        dhr_total_loss = (dhr_real_loss + dhr_fake_loss) * 0.5
        dlr_real_loss = self.adv_loss(ones_like(real_lr_pred), real_lr_pred)
        dlr_fake_loss = self.adv_loss(zeros_like(fake_lr_pred), fake_lr_pred)
        dlr_total_loss = (dlr_real_loss + dlr_fake_loss) * 0.5
        total_disc_loss = dhr_total_loss + dlr_total_loss

        perceptual_loss = self.perceptual_loss(real_hr, fake_hr)
        ghr_adv = self.adv_loss(ones_like(fake_hr_pred), fake_hr_pred)
        glr_adv = self.adv_loss(ones_like(fake_lr_pred), fake_lr_pred)
        adv_loss = ghr_adv + glr_adv
        tv_loss = self.tv_loss(real_hr, fake_hr)
        content_loss = self.content_loss(real_hr, fake_hr)
        cyc_forward = self.cyc_loss(cycled_lr, real_lr)
        cyc_backward = self.cyc_loss(cycled_hr, real_hr)
        cyc_loss = cyc_forward + cyc_backward
        total_gen_loss = perceptual_loss + adv_loss + tv_loss + \
                        (content_loss * self.lambda_content) + \
                        (cyc_loss * self.lambda_cyc)

        # za progress bar
        return {
            "Total discriminator loss": total_disc_loss,
            "Total generator loss": total_gen_loss
        }

    def call(self, inputs, training=False):
        """
        Ob klicu generira sliko visoke resolucije.
        """
        return self.G_HR(inputs, training=training)
