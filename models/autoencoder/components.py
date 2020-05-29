import math

import numpy as np
import tensorflow as tf
import tensorflow_addons

import models.io
import util


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
    # ^ Recall that we are feeding in images normalized to [-1, 1].
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
    filter_growth=1,
    strides=1,
    dilation_rate=2,
    use_sampling=True,
    padding='same'
):
    inputs = tf.keras.Input(shape=image_shape, name='encoder_input')
    x = inputs
    # Dirty trick to avoid odd numbers on the way.
    factor = 2 ** levels
    if image_shape[0] % factor != 0 or image_shape[1] % factor != 0:
        extend_h = (math.ceil(image_shape[0] / factor) * factor - image_shape[0]) // 2
        extend_w = (math.ceil(image_shape[1] / factor) * factor - image_shape[1]) // 2
        x = util.ReflectionPadding2D((extend_w, extend_h))(x)
    for _ in range(levels):
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation=leaky_relu,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding
        )(x)
        if use_sampling:
            x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
        filters = int(filters * filter_growth)

    # Shape needed to build decoder model.
    # Could also be calculated manually, but this is less error prone.
    last_conv_shape = tf.keras.backend.int_shape(x)[1:]

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(intermediate_dim, activation=leaky_relu)(x)
    latent_outputs = \
        tf.keras.layers.Dense(encoding_dim, name='z', activation=leaky_relu)(x)
    encoder = tf.keras.Model(inputs, latent_outputs, name='encoder')
    return encoder, last_conv_shape


def cnn_decoder(
    image_shape,
    encoding_dim,
    last_encoder_conv_shape,
    intermediate_dim,
    levels=1,
    kernel_size=3,
    filter_growth=1,
    strides=1,
    dilation_rate=2,
    use_sampling=True,
    padding='same'
):
    latent_inputs = tf.keras.Input(shape=(encoding_dim,), name='z')
    x = tf.keras.layers.Dense(
        intermediate_dim, activation=leaky_relu
    )(latent_inputs)
    x = tf.keras.layers.Dense(np.prod(last_encoder_conv_shape), activation=leaky_relu)(x)
    x = tf.keras.layers.Reshape(last_encoder_conv_shape)(x)
    filters = last_encoder_conv_shape[-1]
    for _ in range(levels - 1):
        filters = int(filters / filter_growth)
        if use_sampling:
            x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            activation=leaky_relu,
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
    # ^ Recall that we are feeding in images normalized to [-1, 1].
    # Dirty trick to avoid odd numbers on the way. Invert padding in encoder.
    factor = 2 ** levels
    if image_shape[0] % factor != 0 or image_shape[1] % factor != 0:
        extend_h = (math.ceil(image_shape[0] / factor) * factor - image_shape[0]) // 2
        extend_w = (math.ceil(image_shape[1] / factor) * factor - image_shape[1]) // 2
        outputs = outputs[
            :,
            extend_h:image_shape[0] + extend_h,
            extend_w:image_shape[1] + extend_w,
            :
        ]
    decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')
    return decoder


def adain_encoder(
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
    "Architecture inspired by StyleGAN https://arxiv.org/abs/1812.04948."
    inputs = tf.keras.Input(shape=image_shape, name='encoder_input')
    moments = []
    x = inputs
    # Dirty trick to avoid odd numbers on the way.
    factor = 2 ** levels
    if image_shape[0] % factor != 0 or image_shape[1] % factor != 0:
        extend_h = (math.ceil(image_shape[0] / factor) * factor - image_shape[0]) // 2
        extend_w = (math.ceil(image_shape[1] / factor) * factor - image_shape[1]) // 2
        x = util.ReflectionPadding2D((extend_w, extend_h))(x)
    for _ in range(levels):
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation=leaky_relu,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding
        )(x)
        moments.append(InstanceMoments()(x))
        if use_sampling:
            x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
        filters *= 2

    # Shape needed to build decoder model.
    # Could also be calculated manually, but this is less error prone.
    last_conv_shape = tf.keras.backend.int_shape(x)[1:]

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.concatenate([x] + moments)
    x = tf.keras.layers.Dense(intermediate_dim, activation=leaky_relu)(x)
    latent_outputs = \
        tf.keras.layers.Dense(encoding_dim, name='z', activation=leaky_relu)(x)
    encoder = tf.keras.Model(inputs, latent_outputs, name='encoder')
    return encoder, last_conv_shape


def adain_decoder(
    image_shape,
    encoding_dim,
    intermediate_dim,
    last_encoder_conv_shape,
    levels=1,
    kernel_size=3,
    strides=1,
    dilation_rate=2,
    use_sampling=True,
    padding='same'
):
    "Architecture inspired by StyleGAN https://arxiv.org/abs/1812.04948."
    latent_inputs = tf.keras.Input(shape=(encoding_dim,), name='z')
    x_intermediate = tf.keras.layers.Dense(
        intermediate_dim, activation='relu'
    )(latent_inputs)
    x = tf.keras.layers.Dense(
        np.prod(last_encoder_conv_shape), activation='relu'
    )(x_intermediate)
    x = tf.keras.layers.Reshape(last_encoder_conv_shape)(x)
    filters = last_encoder_conv_shape[-1]
    for _ in range(levels - 1):
        filters //= 2
        if use_sampling:
            x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = adaptive_instance_normalization(
            x.shape[1:], x_intermediate.shape[1:]
        )([x, x_intermediate])
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
    x = adaptive_instance_normalization(
            x.shape[1:], x_intermediate.shape[1:]
        )([x, x_intermediate])
    outputs = tf.keras.layers.Conv2DTranspose(
        filters=image_shape[-1],
        kernel_size=kernel_size,
        activation='tanh',
        strides=strides,
        dilation_rate=dilation_rate,
        padding=padding,
        name='decoder_output'
    )(x)
    # ^ Recall that we are feeding in images normalized to [-1, 1].
    # Dirty trick to avoid odd numbers on the way. Invert padding in encoder.
    factor = 2 ** levels
    if image_shape[0] % factor != 0 or image_shape[1] % factor != 0:
        extend_h = (math.ceil(image_shape[0] / factor) * factor - image_shape[0]) // 2
        extend_w = (math.ceil(image_shape[1] / factor) * factor - image_shape[1]) // 2
        outputs = outputs[
            :,
            extend_h:image_shape[0] + extend_h,
            extend_w:image_shape[1] + extend_w,
            :
        ]
    decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')
    return decoder


def adaptive_instance_normalization(
    feature_maps_shape,
    latent_inputs_shape
):
    "Original idea from https://arxiv.org/abs/1703.06868."
    latent_inputs = tf.keras.layers.Input(shape=latent_inputs_shape)
    feature_maps = tf.keras.layers.Input(shape=feature_maps_shape)
    mean = tf.keras.layers.Dense(feature_maps.shape[-1])(latent_inputs)
    variance = tf.keras.layers.Dense(feature_maps.shape[-1])(latent_inputs)
    normalized = tensorflow_addons.layers.InstanceNormalization()(feature_maps)
    outputs = tf.keras.layers.add([
        tf.keras.layers.multiply([normalized, variance]),
        mean
    ])
    return tf.keras.Model(
        inputs=[feature_maps, latent_inputs],
        outputs=outputs,
    )


class InstanceMoments(tf.keras.layers.Layer):
    """Compute instance moments, i.e. moments for each channel.
    Basically a wrapper for tf.nn.moments(x, axes=[1, 2])
    """
    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            2 * input_shape[3]
        )

    def call(self, inputs, **kwargs):
        return tf.concat(
            tf.nn.moments(inputs, axes=[1, 2]),
            axis=1
        )


def leaky_relu(x):
    return tf.keras.activations.relu(x, alpha=0.2)


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
