"""Components used in the CycleGAN model.

See section 7.2. Network architectures in https://arxiv.org/pdf/1703.10593.pdf
regarding naming conventions and meaning of the model blocks.
"""

import random

import numpy as np
import tensorflow as tf

import config
import util


def multi_scale_discriminator(
    image_shape,
    normalization,
    use_patchgan_discriminator,
    name=None
):
    x1 = tf.keras.Input(shape=image_shape)
    x2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x1)
    out_x1 = discriminator(
        image_shape,
        normalization,
        use_patchgan_discriminator,
        'D1'
    )(x1)
    out_x2 = discriminator(
        image_shape,
        normalization,
        use_patchgan_discriminator,
        'D2'
    )(x2)
    return tf.keras.Model(inputs=x1, outputs=[out_x1, out_x2], name=name)


def discriminator(
    image_shape,
    normalization,
    use_patchgan_discriminator,
    name=None
):
    # Specify input
    input_img = tf.keras.Input(shape=image_shape)
    # Layer 1 (instance normalization is not used for this layer)
    x = ck(input_img, 64, 2)
    # Layer 2
    x = ck(x, 128, 2, normalization)
    # Layer 3
    x = ck(x, 256, 2, normalization)
    # Layer 4
    x = ck(x, 512, 1, normalization)
    # Output layer
    if use_patchgan_discriminator:
        x = tf.keras.layers.Conv2D(
            filters=1, kernel_size=4, strides=1, padding='same'
        )(x)
    else:
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)
    # No sigmoid (LSGAN) to avoid near-fp32 machine epsilon discriminator cost.
    return tf.keras.Model(inputs=input_img, outputs=x, name=name)


def generator(
    image_shape,
    normalization,
    use_multiscale_discriminator,
    use_resize_convolution,
    name=None
):
    # Specify input
    input_img = tf.keras.Input(shape=image_shape)
    # Layer 1
    x = util.ReflectionPadding2D((3, 3))(input_img)
    x = c7Ak(x, 32, normalization)
    # Layer 2
    x = dk(x, 64, normalization)
    # Layer 3
    x = dk(x, 128, normalization)
    # Layer 3.5
    if use_multiscale_discriminator:
        x = dk(x, 256, normalization)
    # Layer 4-12: Residual layer
    for _ in range(4, 13):
        x = Rk(x, normalization)
    # Layer 12.5
    if use_multiscale_discriminator:
        x = uk(
            x, 128,
            normalization, use_resize_convolution
        )
    # Layer 13
    x = uk(
        x, 64,
        normalization, use_resize_convolution
    )
    # Layer 14
    x = uk(
        x, 32,
        normalization, use_resize_convolution
    )
    x = util.ReflectionPadding2D((3, 3))(x)
    x = tf.keras.layers.Conv2D(
        image_shape[-1], kernel_size=7, strides=1
    )(x)
    x = tf.keras.activations.tanh(x)
    # ^ They say they use ReLU, but they don't.
    return tf.keras.Model(inputs=input_img, outputs=x, name=name)


def ck(x, k, stride, normalization=None):
    x = tf.keras.layers.Conv2D(
        filters=k, kernel_size=4, strides=stride, padding='same'
    )(x)
    # Normalization is not done on the first discriminator layer
    if normalization:
        x = config.NORMALIZATIONS[normalization](
            axis=3, center=True, epsilon=1e-5
        )(x, training=True)
    x = tf.keras.activations.relu(x, alpha=0.2)  # LeakyReLU
    return x


def c7Ak(x, k, normalization):
    x = tf.keras.layers.Conv2D(
        filters=k, kernel_size=7, strides=1, padding='valid'
    )(x)
    x = config.NORMALIZATIONS[normalization](
        axis=3, center=True, epsilon=1e-5
    )(x, training=True)
    x = tf.keras.activations.relu(x)
    return x


def dk(x, k, normalization):
    x = tf.keras.layers.Conv2D(
        filters=k, kernel_size=3, strides=2, padding='same'
    )(x)
    x = config.NORMALIZATIONS[normalization](
        axis=3, center=True, epsilon=1e-5
    )(x, training=True)
    x = tf.keras.activations.relu(x)
    return x


def Rk(x0, normalization):
    k = int(x0.shape[-1])
    # first layer
    x = util.ReflectionPadding2D((1, 1))(x0)
    x = tf.keras.layers.Conv2D(
        filters=k, kernel_size=3, strides=1, padding='valid'
    )(x)
    x = config.NORMALIZATIONS[normalization](
        axis=3, center=True, epsilon=1e-5
    )(x, training=True)
    x = tf.keras.activations.relu(x)
    # second layer
    x = util.ReflectionPadding2D((1, 1))(x)
    x = tf.keras.layers.Conv2D(
        filters=k, kernel_size=3, strides=1, padding='valid'
    )(x)
    x = config.NORMALIZATIONS[normalization](
        axis=3, center=True, epsilon=1e-5
    )(x, training=True)
    # merge
    x = tf.keras.layers.add([x, x0])
    return x


def uk(x, k, normalization, use_resize_convolution):
    # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
    if use_resize_convolution:
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)  # Nearest neighbour
        x = util.ReflectionPadding2D((1, 1))(x)
        x = tf.keras.layers.Conv2D(
            filters=k, kernel_size=3, strides=1, padding='valid'
        )(x)
    else:
        x = tf.keras.layers.Conv2DTranspose(
            filters=k, kernel_size=3, strides=2, padding='same'
        )(x)  # this matches fractionally stided with stride 1/2
    x = config.NORMALIZATIONS[normalization](
        axis=3, center=True, epsilon=1e-5
    )(x, training=True)
    x = tf.keras.activations.relu(x)
    return x


class ImagePool():
    "Image pools used to update the discriminators."
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            if len(image.shape) == 3:
                image = image[np.newaxis, :, :, :]

            if self.num_imgs < self.pool_size:  # fill up the image pool
                self.num_imgs = self.num_imgs + 1
                if len(self.images) == 0:
                    self.images = image
                else:
                    self.images = np.vstack((self.images, image))

                if len(return_images) == 0:
                    return_images = image
                else:
                    return_images = np.vstack((return_images, image))

            else:
                # 50% chance that we return and replace an old synthetic image
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id, :, :, :]
                    tmp = tmp[np.newaxis, :, :, :]
                    self.images[random_id, :, :, :] = image[0, :, :, :]
                    if len(return_images) == 0:
                        return_images = tmp
                    else:
                        return_images = np.vstack((return_images, tmp))
                else:
                    if len(return_images) == 0:
                        return_images = image
                    else:
                        return_images = np.vstack((return_images, image))

        return return_images
