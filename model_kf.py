import einops
import keras
from keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import model_to_dot
from IPython.display import display, Latex


class Conv2Plus1D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding):
        """
        A sequence of convolutional layers that first apply the convolution operation over the
        spatial dimensions, and then the temporal dimension.
        """
        super().__init__()
        self.seq = keras.Sequential(
            [
                # Spatial decomposition
                layers.Conv3D(
                    filters=filters,
                    kernel_size=(1, kernel_size[1], kernel_size[2]),
                    padding=padding,
                ),
                # Temporal decomposition
                layers.Conv3D(
                    filters=filters, kernel_size=(kernel_size[0], 1, 1), padding=padding
                ),
            ]
        )

    def call(self, x):
        return self.seq(x)

    # def get_config(self):
    #   # Serialize the configuration of your layer
    #   config = {
    #       'filters': self.filters,
    #       'kernel_size': self.kernel_size,
    #       'padding': self.padding
    #   }
    #   base_config = super(Conv2Plus1D, self).get_config()
    #   return dict(list(base_config.items()) + list(config.items()))


class ResidualMain(keras.layers.Layer):
    """
    Residual block of the model with convolution, layer normalization, and the
    activation function, ReLU.
    """

    def __init__(self, filters, kernel_size):
        super().__init__()
        self.seq = keras.Sequential(
            [
                Conv2Plus1D(filters=filters, kernel_size=kernel_size, padding="same"),
                layers.LayerNormalization(),
                layers.ReLU(),
                Conv2Plus1D(filters=filters, kernel_size=kernel_size, padding="same"),
                layers.LayerNormalization(),
            ]
        )

    def call(self, x):
        return self.seq(x)


class Project(keras.layers.Layer):
    """
    Project certain dimensions of the tensor as the data is passed through different
    sized filters and downsampled.
    """

    def __init__(self, units):
        super().__init__()
        self.seq = keras.Sequential([layers.Dense(units), layers.LayerNormalization()])

    def call(self, x):
        return self.seq(x)


def add_residual_block(input, filters, kernel_size):
    """
    Add residual blocks to the model. If the last dimensions of the input data
    and filter size does not match, project it such that last dimension matches.
    """
    out = ResidualMain(filters, kernel_size)(input)

    res = input
    # Using the Keras functional APIs, project the last dimension of the tensor to
    # match the new filter size
    if out.shape[-1] != input.shape[-1]:
        res = Project(out.shape[-1])(res)

    return layers.add([res, out])


class ResizeVideo(keras.layers.Layer):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.resizing_layer = layers.Resizing(self.height, self.width)

    def call(self, video):
        """
        Use the einops library to resize the tensor.

        Args:
          video: Tensor representation of the video, in the form of a set of frames.

        Return:
          A downsampled size of the video according to the new height and width it should be resized to.
        """
        # b stands for batch size, t stands for time, h stands for height,
        # w stands for width, and c stands for the number of channels.
        old_shape = einops.parse_shape(video, "b t h w c")
        images = einops.rearrange(video, "b t h w c -> (b t) h w c")
        images = self.resizing_layer(images)
        videos = einops.rearrange(images, "(b t) h w c -> b t h w c", t=old_shape["t"])
        return videos


def create_model(n_frames, height, width):
    input_shape = (None, n_frames, height, width, 3)
    input = layers.Input(shape=(input_shape[1:]))
    x = input

    x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = ResizeVideo(height // 2, width // 2)(x)

    # Block 1
    x = add_residual_block(x, 16, (3, 3, 3))
    x = ResizeVideo(height // 4, width // 4)(x)

    # Block 2
    x = add_residual_block(x, 32, (3, 3, 3))
    x = ResizeVideo(height // 8, width // 8)(x)

    # Block 3
    x = add_residual_block(x, 64, (3, 3, 3))
    x = ResizeVideo(height // 16, width // 16)(x)

    # Block 4
    x = add_residual_block(x, 128, (3, 3, 3))

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Flatten()(x)
    # this was 10. but might be causing an issue as it is a binary categorisation task now
    x = layers.Dense(4)(x)
    # x = layers.Dense(1, activation="sigmoid")(x) # Change for binary classification

    return keras.Model(input, x)

