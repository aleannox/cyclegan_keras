import numpy as np
import tensorflow as tf

import models.io


def mlp_encoder(
    image_shape,
    encoding_dim,
    intermediate_dim,
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
    intermediate_dim
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
    intermediate_dim,
    levels=1,
    kernel_size=3,
    filters=4,
    strides=1,
    dilation_rate=2,
    use_sampling=True,
    padding='same'
):
    inputs = tf.keras.Input(shape=image_shape, name='encoder_input')
    x = inputs
    for _ in range(levels):
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation='relu',
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding
        )(x)
        if use_sampling:
            x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
        filters /= 2

    # Shape needed to build decoder model.
    # Could also be calculated manually, but this is less error prone.
    last_conv_shape = tf.keras.backend.int_shape(x)[1:]

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(intermediate_dim, activation='relu')(x)
    latent_outputs = tf.keras.layers.Dense(encoding_dim, name='z')(x)
    encoder = tf.keras.Model(inputs, latent_outputs, name='encoder')
    return encoder, last_conv_shape


def cnn_decoder(
    image_shape,
    encoding_dim,
    last_encoder_conv_shape,
    intermediate_dim,
    levels=1,
    kernel_size=3,
    strides=1,
    dilation_rate=2,
    use_sampling=True,
    padding='same'
):
    latent_inputs = tf.keras.Input(shape=(encoding_dim,), name='z')
    x = tf.keras.layers.Dense(
        intermediate_dim, activation='relu'
    )(latent_inputs)
    x = tf.keras.layers.Dense(np.prod(last_encoder_conv_shape), activation='tanh')(x)
    x = tf.keras.layers.Reshape(last_encoder_conv_shape)(x)
    filters = last_encoder_conv_shape[-1]
    for _ in range(levels - 1):
        filters *= 2
        if use_sampling:
            x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            activation='relu',
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding
        )(x)
    if use_sampling:
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    outputs = tf.keras.layers.Conv2DTranspose(
        filters=image_shape[-1],
        kernel_size=kernel_size,
        activation='tanh',
        strides=strides,
        dilation_rate=dilation_rate,
        padding=padding,
        name='decoder_output'
    )(x)
    decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')
    return decoder


class SaveExamplesCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        save_interval,
        real_examples,
        sink_folders,
        num_channels
    ):
        self.save_interval = save_interval
        self.real_examples = real_examples
        self.sink_folders = sink_folders
        self.num_channels = num_channels
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        for train_or_test in ['train', 'test']:
            real_images = self.real_examples[train_or_test]
            reconstructed_images = self.model.predict(real_images)
            for i in range(len(real_images)):
                models.io.save_image_tuple(
                    (
                        real_images[i],
                        reconstructed_images[i]
                    ),
                    self.sink_folders[train_or_test] /
                    f'epoch{epoch:04d}_example{i}.jpg',
                    self.num_channels
                )
