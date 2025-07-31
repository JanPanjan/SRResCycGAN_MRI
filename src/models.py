from keras import Model, layers
from utils import LR_SHAPE, HR_SHAPE


def Generator_HR(input_shape=LR_SHAPE):
    """ Generator za HR (high resolution) slike.

    Deluje na principu "upscale -> refine". Sprejme sliko nizke ločljivosti in
    vrne izboljšano sliko visok ločljivosti. Arhitektura temelji na članku o SRResCGAN.
    Ima 3 glavne dele:

    1. Encoder - upsampling vhodne slike in ekstrakcija značilnosti
    2. ResidualNet - predela zemljevid značilnosti (feature map) od encoderja
    3. Decoder - na podalagi izhoda ResNet-a ustvari zemljevid napak (proximal map)

    `Subtract` sloj zmanjša prisotnost šuma in artefaktov upsampled slike s pomočjo
    zemljevida napak. Zadnji `Conv2D` sloj poskrbi, da je izhod slika s tremi kanali.

    Args:
        input_shape: dimenzija LR slike (width, height, channels)
    Returns:
        HR slika dimenzije (320, 320, 3)
    """
    FILTERS = 64

    lr_input = layers.Input(shape=input_shape)
    encoder_out = layers.Conv2DTranspose(
        filters=FILTERS,     # število kanalov izhoda; rgb slika ima 3, tu jih vrne 64)
        kernel_size=5,  # dimenzije jedra, ki bo procesiral sliko (5x5)
        strides=4,      # za koliko pikslov se premakne jedro (4 => 4x upsampling => vodi v 320x320), ko je padding="same"
        padding="same"  # poskrbi, da se doda ravno prav ničel na robove, da je končna dimenzija odvisna od strides
    )(lr_input)

    global_skip = encoder_out

    # služi kot vhod in izhod ResNet-a.
    x = layers.Conv2D(filters=FILTERS, kernel_size=5, padding="same")(encoder_out)

    for _ in range(5):
        block_skip = x
        x = layers.PReLU(shared_axes=[1, 2])(block_skip)
        x = layers.Conv2D(filters=FILTERS, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
        x = layers.GroupNormalization(groups=-1, axis=-1)(x)  # -1 za InstanceNormalization

        x = layers.PReLU(shared_axes=[1, 2])(x)
        x = layers.Conv2D(filters=FILTERS, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
        x = layers.GroupNormalization(groups=-1, axis=-1)(x)
        x = layers.Add()([block_skip, x])

    decoder_out = layers.Conv2D(filters=FILTERS, kernel_size=5, strides=1, padding="same")(x)
    decoder_out = layers.Conv2D(filters=FILTERS, kernel_size=5, strides=1, padding="same")(decoder_out)

    subtracted = layers.Subtract()([global_skip, decoder_out])
    model_out = layers.Conv2D(filters=3, kernel_size=3, strides=1, padding="same", use_bias=False)(subtracted)
    model_out = layers.ReLU(max_value=255)(model_out)  # vrednosti spravi na interval [0,255]

    return Model(inputs=lr_input, outputs=model_out, name="G_HR")


def Generator_LR(input_shape=HR_SHAPE):
    """ Generator za LR (low resolution) slike.

    Sprejme sliko visoke ločljivosti in vrne degradirano sliko nizke ločljivosti. Arhitektura temelji
    na G3 generatorju iz CinCGAN strukture, ki opravi downsampling s pomočjo konvolucije.
    Ima 3 glavne dele:

    1. Glava - downsampling in ekstrakcija značilnosti
    2. ResidualNet - predela izdelan zemljevid značilnosti (feature map)
    3. Rep - transformacija nazaj v sliko s tremi kanali

    Args:
        input_shape: dimenzija HR slike (width, height, channels)
    Returns:
        HR slika dimenzije (320, 320, 3)
    """
    FILTERS = 64

    hr_input = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=FILTERS, kernel_size=7, padding="same")(hr_input)
    x = layers.GroupNormalization(groups=-1, axis=-1)(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    for _ in range(2):  # downsample
        x = layers.Conv2D(filters=FILTERS, kernel_size=3, strides=2, padding="same")(x)
        x = layers.GroupNormalization(groups=-1, axis=-1)(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)

    for _ in range(6): # ResNet
        block_skip = x
        x = layers.Conv2D(filters=FILTERS, kernel_size=3, padding="same")(x)
        x = layers.GroupNormalization(groups=-1, axis=-1)(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)

        x = layers.Conv2D(filters=FILTERS, kernel_size=3, padding="same")(x)
        x = layers.GroupNormalization(groups=-1, axis=-1)(x)
        x = layers.Add()([block_skip, x])
        x = layers.LeakyReLU(negative_slope=0.2)(x)

    for _ in range(2):
        x = layers.Conv2D(filters=FILTERS, kernel_size=3, padding="same")(x)
        x = layers.GroupNormalization(groups=-1, axis=-1)(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)

    x = layers.Conv2D(filters=3, kernel_size=7, padding="same")(x)
    model_out = layers.ReLU(max_value=255)(x)  # vrednosti spravi na interval [0,255]

    return Model(inputs=hr_input, outputs=model_out, name="G_LR")


def Discriminator_HR(input_shape=HR_SHAPE):
    """ Diskriminator za HR (high resolution) slike.

    Sprejme HR sliko (pravo ali lažno) in vrne verjetnost,
    da je slika prava. Deluje kot binarni klasifikator.
    Arhitektura temelji na članku o SRResCGAN. Ima 3 glavne dele:

    1. Vhodni konvolucijski blok
    2. Zaporedje konvolucjiskih blokov, ki
       - postopoma zmanjšujejo dimenzijo slike,
       - normalizirajo vrednosti za stabilizacijo pri treniranju
       - večajo število filtrov za procesiranje lastnosti slike
    3. Fully connected sloji, ki poskrbijo, da se vrne ena float vrednost

    Args:
        input_shape: dimenzija HR slike (width, height, channels)
    Returns:
        float logit vrednost
    """
    hr_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="valid")(hr_input)
    x = layers.LeakyReLU()(x)

    KERNEL_SIZE = (4, 3)
    STRIDES = (2, 1)
    FILTERS = (
        64,
        128, 128,
        256, 256,
        512, 512, 512, 512
    )

    for i in range(9):
        kernel_size = KERNEL_SIZE[0] if i % 2 == 0 else KERNEL_SIZE[1]
        strides = STRIDES[0] if i % 2 == 0 else STRIDES[1]

        x = layers.Conv2D(filters=FILTERS[i], kernel_size=kernel_size, strides=strides, padding="valid")(x)
        # TODO: Spectral Normalization?
        x = layers.GroupNormalization(groups=-1, axis=-1)(x)
        x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(units=100)(x)
    x = layers.LeakyReLU()(x)
    model_out = layers.Dense(units=1)(x)

    return Model(inputs=hr_input, outputs=model_out, name="D_HR")


def Discriminator_LR(input_shape=LR_SHAPE):
    """ Diskriminator za LR (low resolution) slike.

    Arhitektura temelji PatchGAN diskriminatorju, kot opisujejo v članku o SRResCycGAN, ki vrne matriko
    ocen verjetnosti (logitov) namesto ene same vrednosti. Tak pristop spodbuja generator, da se osredotoči
    na ustvarjanje realističnih lokalnih podrobnosti. Namesto BatchNormalization je uporabljena
    InstanceNormalization, saj se obnese boljše med treniranjem.

    Args:
        input_shape: dimenzija LR slike (width, height, channels)
    Returns:
        matrika logitov oblike (20, 20, 1)
    """
    KERNEL_SIZE = 5
    FILTERS = [64, 128, 256]

    lr_input = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=FILTERS[0], kernel_size=KERNEL_SIZE, strides=2, padding="same")(lr_input)
    x = layers.GroupNormalization(groups=-1, axis=-1)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(filters=FILTERS[1], kernel_size=KERNEL_SIZE, strides=2, padding="same")(x)
    x = layers.GroupNormalization(groups=-1, axis=-1)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(filters=FILTERS[2], kernel_size=KERNEL_SIZE, padding="same")(x)
    x = layers.GroupNormalization(groups=-1, axis=-1)(x)
    x = layers.LeakyReLU()(x)

    model_out = layers.Conv2D(filters=1, kernel_size=KERNEL_SIZE, padding="same")(x)

    return Model(inputs=lr_input, outputs=model_out, name="D_LR")