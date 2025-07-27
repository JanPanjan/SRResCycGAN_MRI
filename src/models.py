from keras import Model, layers


def Generator_HR(input_shape=(80, 80, 3)):
    """ Generator za HR (high resolution) slike.

    Deluje na principu "upscale -> refine". Sprejme sliko nizke ločljivosti in
    vrne izboljšano sliko visok ločljivosti. Arhitektura temelji na članku o SRResCGAN.
    Ima 3 glavne dele:

    1. Encoder - poveča resolucijo vhodne slike in ustvari feature map s 64 kanali
    2. ResidualNet - predela feature map od encoderja
    3. Decoder - na podalagi izhoda ResNet-a ustvari proximal map (zemljevid napak)

    `Substract` na koncu izloči te zaznane napake iz povečane slike.

    ResNet je sestavljen iz 5 blokov. Na koncu vsakega bloka se združijo vrednosti med vhodnimi in izhodnimi podatki
    (od trenutnega bloka) preko skip povezave. Ta izhod služi kot vhod v naslednji blok.

    Args:
        input_shape: dimenzija LR slike (width, height, channels)
    Returns:
        HR slika dimenzije (320, 320, 3)
    """
    lr_input = layers.Input(shape=input_shape)
    encoder_out = layers.Conv2DTranspose(
        filters=64,     # število kanalov izhoda; rgb slika ima 3, tu jih vrne 64)
        kernel_size=5,  # dimenzije jedra, ki bo procesiral sliko (5x5)
        strides=4,      # za koliko pikslov se premakne jedro (4 => 4x upsampling => vodi v 320x320), ko je padding="same"
        padding="same"  # poskrbi, da se doda ravno prav ničel na robove, da je končna dimenzija odvisna od strides
    )(lr_input)

    global_skip = encoder_out

    # služi kot vhod in izhod ResNet-a.
    resnet_skip = layers.Conv2D(filters=64, kernel_size=5, padding="same")(encoder_out)

    for _ in range(5):
        block_in = resnet_skip
        x = layers.PReLU(shared_axes=[1, 2])(block_in)
        x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
        x = layers.PReLU(shared_axes=[1, 2])(x)
        x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
        resnet_skip = layers.Add()([block_in, x])

    decoder_out = layers.Conv2D(filters=64, kernel_size=5, strides=1, padding="same")(resnet_skip)
    decoder_out = layers.Conv2D(filters=64, kernel_size=5, strides=1, padding="same")(decoder_out)

    subtracted = layers.Subtract()([global_skip, decoder_out])
    model_out = layers.Conv2D(filters=3, kernel_size=3, strides=1, padding="same", use_bias=False)(subtracted)
    model_out = layers.ReLU(max_value=255)(model_out)  # vrednosti spravi na interval [0,255]

    return Model(inputs=lr_input, outputs=model_out, name="G_HR")


def Discriminator_HR(input_shape=(320, 320, 3)):
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
        float verjetnost
    """
    hr_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="valid")(hr_input)
    x = layers.LeakyReLU()(x)

    KERNEL_SIZE = (4, 3)
    STRIDES = (2, 1)
    FILTERS = (
        1 * (64),
        2 * (128),
        2 * (256),
        4 * (512)
    )

    for i in range(9):
        kernel_size = KERNEL_SIZE[0] if i % 2 == 0 else KERNEL_SIZE[1]
        strides = STRIDES[0] if i % 2 == 0 else STRIDES[1]

        x = layers.Conv2D(filters=FILTERS[i], kernel_size=kernel_size, strides=strides, padding="valid")(x)
        # TODO: Spectral Normalization?
        x = layers.GroupNormalization(groups=-1, axis=-1)(x)  # -1 za InstanceNormalization in zadnjo dimenzijo (kanali)
        x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(units=100)(x)
    x = layers.LeakyReLU()(x)
    model_out = layers.Dense(units=1)(x)

    return Model(inputs=hr_input, outputs=model_out, name="D_HR")