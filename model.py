
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    Input,
    MaxPool2D,
    ZeroPadding2D,
    Flatten,
    Dense,
    ReLU,
    BatchNormalization,
    Activation,
)

from tensorflow.keras.regularizers import l2

trainable = False


def ConvLayer(
    x,
    filters,
    size,
    strides=1,
    padding="same",
    bias=False,
    batch_norm=True,
    batch_size=1,
    l2_reg=0.0005,
    name="layer",
):
    """
    Apply a convolution layer to the input tensor 'x' with L2 regularization and optional 
    batch normalization.
    """

    x = Conv2D(
        filters=filters,
        kernel_size=size,
        strides=strides,
        padding=padding,
        use_bias=bias,
        kernel_regularizer=l2(l2_reg),
        name=name,
    )(x)

    if batch_norm:
        x = BatchNormalization(trainable=trainable, name="bnorm_%s" % name)(x)
        x = ReLU()(x)

    return x


def ConvLayerAlternate(
    x,
    filters,
    size,
    strides=1,
    padding="same",
    bias=False,
    batch_norm=True,
    l2_reg=0.0005,
):
    """
    An alternate version of the ConvLayer function with different batch normalization 
    momentum and an option for 'valid' padding with adjusted padding.
    """

    func = {
        "relu": ReLU(),
    }

    if padding == "valid":
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding

    x = Conv2D(
        filters=filters,
        kernel_size=size,
        strides=strides,
        padding=padding,
        use_bias=bias,
        kernel_regularizer=l2(l2_reg),
    )(x)

    if batch_norm:
        x = BatchNormalization(momentum=0.5)(x)
        x = ReLU(x)

    return x


def NeuralNetwork(
    name="example",
    input_shape=(256, 256),
    output_activation="sigmoid",
    minibatch_size=32,
    dropout_reg=0.5,
    l2_reg=0.0005,
    is_training=False,
):
    global trainable
    trainable = is_training

    x = inputs = Input([input_shape[0], input_shape[1], 3])
    layers = {}

    layers["inputs"] = ConvLayer(
        inputs,
        4,
        1,
        strides=1,
        batch_size=minibatch_size,
        name="inputs",
    )

    layers["examplenet1"] = ConvLayer(
        layers["inputs"],
        4,
        3,
        strides=2,
        batch_size=minibatch_size,
        name="examplenet1",
    )

    layers["examplenet2"] = ConvLayer(
        layers["examplenet1"],
        32,
        3,
        strides=2,
        batch_size=minibatch_size,
        name="examplenet2",
    )

    layers["examplenet3"] = ConvLayer(
        layers["examplenet2"],
        16,
        1,
        strides=2,
        batch_size=minibatch_size,
        name="examplenet3",
    )

    layers["examplenet4"] = ConvLayer(
        layers["examplenet3"],
        64,
        3,
        strides=2,
        batch_size=minibatch_size,
        name="examplenet4",
    )

    layers["examplenet5"] = ConvLayer(
        layers["examplenet4"],
        16,
        3,
        strides=2,
        batch_size=minibatch_size,
        name="examplenet5",
    )

    layers["examplenet6"] = ConvLayer(
        layers["examplenet5"],
        4,
        1,
        strides=2,
        batch_size=minibatch_size,
        name="examplenet6",
    )

    flatten = tf.keras.layers.Flatten(name="flatten")(layers["examplenet6"])
    dense_0 = tf.keras.layers.Dense(
        units=64, name="dense0", kernel_regularizer=l2(l2_reg)
    )(flatten)
    dropout = tf.keras.layers.Dropout(dropout_reg)(dense_0)

    layers["dense"] = ReLU()(dropout)

    x = layers["dense"]
    dense = tf.keras.layers.Dense(units=1, name="dense1")(x)
    activate = Activation(output_activation)(dense)

    model = tf.keras.Model(inputs, activate, name=name)
    return layers, inputs, activate, model