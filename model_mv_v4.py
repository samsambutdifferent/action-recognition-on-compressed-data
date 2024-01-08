import einops
import keras
from keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import model_to_dot
from IPython.display import display, Latex

from tensorflow.keras import layers, models


def R2Plus1D_Block(input_tensor, filters, kernel_size):
    # 2D spatial convolution
    x = layers.TimeDistributed(layers.Conv2D(filters, (kernel_size, kernel_size), padding='same', activation='relu'))(input_tensor)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)

    # 1D temporal convolution
    x = layers.Conv3D(filters, (3, 1, 1), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Skip connection
    x = layers.Add()([x, input_tensor]) if input_tensor.shape[-1] == filters else x
    return x


def create_R2Plus1D_mv_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Applying R(2+1)D blocks
    x = R2Plus1D_Block(inputs, 64, 3)
    x = R2Plus1D_Block(x, 64, 3)  # Second block
    x = R2Plus1D_Block(x, 64, 3)  # Third block
    x = R2Plus1D_Block(x, 64, 3)  # Fourth block

    # Flatten and Dense layers
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes)(x)

    model = models.Model(inputs, outputs)
    return model
