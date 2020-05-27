import numpy as np
import tensorflow as tf


def mlp_encoder(
    image_shape,
    encoding_dim,
    intermediate_dim=128,
):
    inputs = tf.keras.Input(shape=image_shape, name='encoder_input')
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(intermediate_dim, activation='relu')(x)
    latent_outputs = tf.keras.layers.Dense(encoding_dim, name='z')(x)
    encoder = tf.keras.Model(inputs, latent_outputs, name='encoder')
    return encoder


def mlp_decoder(
    image_shape,
    encoding_dim,
    intermediate_dim=128
):
    latent_inputs = tf.keras.Input(shape=(encoding_dim,), name='z')
    x = tf.keras.layers.Dense(
        intermediate_dim, activation='relu'
    )(latent_inputs)
    x = tf.keras.layers.Dense(np.prod(image_shape), activation='tanh')(x)
    outputs = tf.keras.layers.Reshape(image_shape)(x)
    decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')
    return decoder


def cnn_encoder(
    image_shape,
    encoding_dim,
    levels=1,
    kernel_size=3,
    filters=4,
    strides=2,
    intermediate_dim=64,
    padding='same'
):
    pass
    # inputs = tf.keras.Input(shape=image_shape, name='encoder_input')
    # x = inputs
    # for _ in range(levels):
    #     filters *= 2
    #     x = tf.keras.layers.Conv2D(
    #         filters=filters,
    #         kernel_size=kernel_size,
    #         activation='relu',
    #         strides=strides,
    #         padding=padding
    #     )(x)

    # # shape info needed to build decoder model
    # shape = tf.keras.backend.int_shape(x)

    # # generate latent vector Q(z|X)
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(intermediate_dim, activation='relu')(x)
    # z_mean = tf.keras.layers.Dense(encoding_dim, name='z_mean')(x)
    # z_log_var = tf.keras.layers.Dense(encoding_dim, name='z_log_var')(x)

def cnn_decoder(
    image_shape,
    encoding_dim,
    levels=1,
    kernel_size=3,
    filters=4,
    strides=2,
    intermediate_dim=64,
    padding='same'
):
    pass
    # latent_inputs = tf.keras.Input(shape=(encoding_dim,), name='z_sampling')
    # x = tf.keras.layers.Dense(np.prod(encoder_o)(latent_inputs)
    # x = tf.keras.layers.Reshape((shape[1], shape[2], shape[3]))(x)
    # for _ in range(levels):
    #     x = tf.keras.layers.Conv2DTranspose(
    #         filters=filters,
    #         kernel_size=kernel_size,
    #         activation='relu',
    #         strides=strides,
    #         padding=padding
    #     )(x)
    #     filters //= 2
    # outputs = tf.keras.layers.Conv2DTranspose(
    #     filters=1,
    #     kernel_size=kernel_size,
    #     activation='sigmoid',
    #     padding=padding,
    #     name='decoder_output'
    # )(x)
