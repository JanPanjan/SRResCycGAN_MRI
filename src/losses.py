from src.utils import HR_SHAPE
from tensorflow import image, reduce_mean, abs
from keras import Loss, Model
from keras.losses import MeanSquaredError, BinaryCrossentropy, MeanAbsoluteError
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input


def total_variation(real_image, fake_image):
    """Total variation loss

    Primerja vodoravne in navpične gradiente resnične in generirane slike. Imenovan tudi
    "gradient difference loss".

    Vrne vsoto razlik gradientov obeh slik.
    """
    real_dy, real_dx = image.image_gradients(real_image)
    fake_dy, fake_dx = image.image_gradients(fake_image)

    loss_dx = reduce_mean(abs(real_dx - fake_dx))
    loss_dy = reduce_mean(abs(real_dy - fake_dy))

    return loss_dx + loss_dy


class PerceptualLoss(Loss):
    """Perceptual loss

    Sliki primerja v prostoru značilnosti (feature space) namesto prostoru slik (image space).
    Feature maps sta dobljena iz nekega vmesnega konvolucijskega sloja VGG19 klasifikatorja.

    Vrne MSE (L2 loss) nad izhodom VGG19 za resnično in generirano sliko.
    """
    def __init__(self, hr_shape=HR_SHAPE) -> None:
        super(PerceptualLoss, self).__init__(name="perceptual_loss")
        self.vgg_model = self.__build_vgg_model(hr_shape)
        self.L2 = MeanSquaredError()


    def __build_vgg_model(self, hr_shape):
        vgg = VGG19(input_shape=hr_shape, include_top=False, weights="imagenet")
        vgg.trainable = False
        output_layer = vgg.get_layer("block3_conv3").output  # Izhod enega od vmesnih slojev
        return Model(inputs=vgg.input, outputs=output_layer, name="vgg_perceptual")


    def call(self, y_true, y_pred):
        true_preproc = preprocess_input(y_true)
        pred_preproc = preprocess_input(y_pred)
        true_feat = self.vgg_model(true_preproc, training=False)
        pred_feat = self.vgg_model(pred_preproc, training=False)
        return self.L2(true_feat, pred_feat)


# -------------------------------------------------------------------------------------

losses_dict = {
    "perceptual": PerceptualLoss(),
    "adversarial": BinaryCrossentropy(from_logits=True),
    "total_variation": total_variation,
    "content": MeanAbsoluteError(),
    "cyclic": MeanAbsoluteError(),
}