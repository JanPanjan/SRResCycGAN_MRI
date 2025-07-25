"""
You load the full VGG model once, and then you define a new
keras.Model that shares the same input as VGG but outputs the
activations from the intermediate layers you're interested in.

Let's build a complete example.

The Code to Create the Feature Extractor

Here is how you would create the feature extractor model. This is a
reusable component in your project.
"""

import keras
import numpy as np


def build_vgg_extractor(layer_names):
    """
    Creates a Keras model that extracts features from the
    specified layers of a VGG19 model.

    Args:
        layer_names (list of str): A list of the names of the
        VGG19 layers to extract.

     Returns:
         tf.keras.Model: A model that takes an image as input
         and returns the feature maps
         from the specified layers.
    """
    # 1. Load the VGG19 model
    # include_top=False means we don't load the final classification layers.
    vgg = keras.applications.VGG19(
        include_top=False,
        weights='imagenet',
        input_shape=(None, None, 3)
    )

    # 2. Freeze the layers
    # We do this to prevent the VGG weights from being updated during GAN training.
    vgg.trainable = False

    # 3. Get the outputs of the desired layers
    # This uses the exact method you proposed: vgg.get_layer(...).output
    outputs = [vgg.get_layer(name).output for name in layer_names]

    # 4. Create the new feature extractor model
    # This model has the same input as VGG but outputs the intermediate activations.
    model = keras.Model(inputs=vgg.input, outputs=outputs,
                        name='vgg_feature_extractor')

    return model


# --- Example Usage ---

# The layer typically used for perceptual loss in recent papers is 'block5_conv4'.
# Let's pick that one.
CONTENT_LAYER = 'block5_conv4'

# Build our feature extractor
feature_extractor = build_vgg_extractor([CONTENT_LAYER])

# Let's see the VGG architecture to find other layer names
# You can uncomment the next line to see all the layer names available
keras.applications.VGG19().summary()

# Create a dummy "fake HR" image to test (Batch, Height, Width, Channels)
# Note: VGG19 expects input pixels in the range [0, 255]
dummy_image = np.random.randint(
    0, 255, size=(1, 256, 256, 3)).astype('float32')
dummy_image.shape

# Pre-process the image for VGG
# `preprocess_input` configures your image to match the exact format that the model was originally trained on.
preprocessed_image = keras.applications.vgg19.preprocess_input(dummy_image)

# Extract the features
features = feature_extractor(preprocessed_image)

print(f"Input image shape: {dummy_image.shape}")
print(f"Extracted features from '{
      CONTENT_LAYER}' have shape: {features.shape}")

"""
How to Use It in Your Loss Function

Now, you can use this feature_extractor model to define your perceptual loss.
"""

# Assume 'feature_extractor' is the model we created above.
# Use Mean Squared Error for the perceptual loss, as it's common.
mse = keras.losses.MeanSquaredError()


def perceptual_loss(real_hr, fake_hr):
    """
    Calculates the perceptual loss between two images.
    """
    # Pre-process images for VGG19
    real_hr_preprocessed = keras.applications.vgg19.preprocess_input(real_hr)
    fake_hr_preprocessed = keras.applications.vgg19.preprocess_input(fake_hr)

    # Extract features
    real_features = feature_extractor(real_hr_preprocessed)
    fake_features = feature_extractor(fake_hr_preprocessed)

    # Calculate the L2 distance (MSE) between the feature maps
    return mse(real_features, fake_features)


# --- Example Usage ---
# Create two dummy images
real_image = np.random.randint(0, 255, size=(1, 256, 256, 3)).astype('float32')
fake_image = np.random.randint(0, 255, size=(1, 256, 256, 3)).astype('float32')

# Calculate the loss
loss = perceptual_loss(real_image, fake_image)

print(f"\nCalculated Perceptual Loss: {loss.numpy()}")

"""
So, to summarize your questions:

* `vgg = keras.applications.VGG19(...)`: Correct.
* `vgg.get_layer(<layer-name>).output`: Correct. This is the key to
    getting the symbolic tensor for the layer's output.
* `vgg.layers`: Correct. You can inspect this property (or better yet,
    vgg.summary()) to see all the layer names you can choose from.

You had all the right pieces. The trick is to assemble them into a new
keras.Model.
"""
