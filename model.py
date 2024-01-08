import einops
import keras
from keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import model_to_dot
from IPython.display import display, Latex


class Conv2Plus1D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding):
        super().__init__()
        self.seq = keras.Sequential(
            [
                layers.Conv3D(
                    filters=filters,
                    kernel_size=(1, kernel_size[1], kernel_size[2]),
                    padding=padding,
                ),
                layers.Conv3D(
                    filters=filters, kernel_size=(kernel_size[0], 1, 1), padding=padding
                ),
            ]
        )

    def call(self, x):
        return self.seq(x)



class ResidualMain(keras.layers.Layer):
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
    def __init__(self, units):
        super().__init__()
        self.seq = keras.Sequential([layers.Dense(units), layers.LayerNormalization()])

    def call(self, x):
        return self.seq(x)


def add_residual_block(input, filters, kernel_size):
    out = ResidualMain(filters, kernel_size)(input)

    res = input
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

    x = add_residual_block(x, 16, (3, 3, 3))
    x = ResizeVideo(height // 4, width // 4)(x)

    x = add_residual_block(x, 32, (3, 3, 3))
    x = ResizeVideo(height // 8, width // 8)(x)

    x = add_residual_block(x, 64, (3, 3, 3))
    x = ResizeVideo(height // 16, width // 16)(x)

    x = add_residual_block(x, 128, (3, 3, 3))

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(4)(x)

    return keras.Model(input, x)
