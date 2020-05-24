import argparse
from collections import OrderedDict
import csv
import datetime
import json
import logging
import os
import random
import sys
import time
import tqdm

import PIL.Image
import numpy as np
from tensorflow.keras.layers import \
    Layer, Input, Conv2D, Activation, add, UpSampling2D, Conv2DTranspose, \
    Flatten, AveragePooling2D, InputSpec, LeakyReLU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model  # pylint: disable=unused-import
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization

import config
import load_data

np.random.seed(seed=12345)


class CycleGAN():
    def __init__(
        self,
        source_images,
        lr_D=2e-4,
        lr_G=2e-4,
        image_shape=(256, 256, 3),
        use_data_generator=False,
        model_key=None,
        generate_synthetic_images=False,
        batch_size=1
    ):
        self.img_shape = image_shape
        self.channels = self.img_shape[-1]
        self.normalization = InstanceNormalization
        # Hyper parameters
        self.lambda_1 = 10.0  # Cyclic loss weight A_2_B
        self.lambda_2 = 10.0  # Cyclic loss weight B_2_A
        self.lambda_D = 1.0  # Weight for loss from discriminator guess on synthetic images
        self.learning_rate_D = lr_D
        self.learning_rate_G = lr_G
        self.generator_iterations = 1  # Number of generator training iterations in each training loop
        self.discriminator_iterations = 1  # Number of generator training iterations in each training loop
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.batch_size = batch_size
        self.epochs = 200  # choose multiples of 25 since the models are save each 25th epoch
        self.save_interval = 1
        self.synthetic_pool_size = 50

        # Linear decay of learning rate, for both discriminators and generators
        self.use_linear_decay = False
        self.decay_epoch = 101  # The epoch where the linear decay of the learning rates start

        # Identity loss - sometimes send images from B to G_A2B (and the opposite) to teach identity mappings
        self.use_identity_learning = False
        self.identity_mapping_modulus = 10  # Identity mapping will be done each time the iteration number is divisable with this number

        # PatchGAN - if false the discriminator learning rate should be decreased
        self.use_patchgan = True

        # Multi scale discriminator - if True the generator have an extra encoding/decoding step to match discriminator information access
        self.use_multiscale_discriminator = False

        # Resize convolution - instead of transpose convolution in deconvolution layers (uk) - can reduce checkerboard artifacts but the blurring might affect the cycle-consistency
        self.use_resize_convolution = False

        # Supervised learning part - for MR images - comparison
        self.use_supervised_learning = False
        self.supervised_weight = 10.0

        # Fetch data during training instead of pre caching all images - might be necessary for large datasets
        self.use_data_generator = use_data_generator

        # Tweaks
        self.REAL_LABEL = 1.0  # Use e.g. 0.9 to avoid training the discriminators to zero loss

        self.result_paths = config.construct_result_paths(model_key)

        self.source_images = source_images

        # optimizer
        self.opt_D = Adam(self.learning_rate_D, self.beta_1, self.beta_2)
        self.opt_G = Adam(self.learning_rate_G, self.beta_1, self.beta_2)

        # ======= Discriminator model ==========
        if self.use_multiscale_discriminator:
            D_A = self.modelMultiScaleDiscriminator()
            D_B = self.modelMultiScaleDiscriminator()
            loss_weights_D = [0.5, 0.5] # 0.5 since we train on real and synthetic images
        else:
            D_A = self.modelDiscriminator()
            D_B = self.modelDiscriminator()
            loss_weights_D = [0.5]  # 0.5 since we train on real and synthetic images
        # D_A.summary()

        # Discriminator builds
        image_A = Input(shape=self.img_shape)
        image_B = Input(shape=self.img_shape)
        guess_A = D_A(image_A)
        guess_B = D_B(image_B)
        self.D_A = Model(inputs=image_A, outputs=guess_A, name='D_A_model')
        self.D_B = Model(inputs=image_B, outputs=guess_B, name='D_B_model')

        # self.D_A.summary()
        # self.D_B.summary()
        self.D_A.compile(optimizer=self.opt_D,
                         loss=self.lse,
                         loss_weights=loss_weights_D)
        self.D_B.compile(optimizer=self.opt_D,
                         loss=self.lse,
                         loss_weights=loss_weights_D)

        self.D_A_static = Model(inputs=image_A, outputs=guess_A, name='D_A_static_model')
        self.D_B_static = Model(inputs=image_B, outputs=guess_B, name='D_B_static_model')

        # ======= Generator model ==========
        # Do note update discriminator weights during generator training
        self.D_A_static.trainable = False
        self.D_B_static.trainable = False

        # Generators
        self.G_A2B = self.modelGenerator(name='G_A2B_model')
        self.G_B2A = self.modelGenerator(name='G_B2A_model')
        # self.G_A2B.summary()

        if self.use_identity_learning:
            self.G_A2B.compile(optimizer=self.opt_G, loss='MAE')
            self.G_B2A.compile(optimizer=self.opt_G, loss='MAE')

        # Generator builds
        real_A = Input(shape=self.img_shape, name='real_A')
        real_B = Input(shape=self.img_shape, name='real_B')
        synthetic_B = self.G_A2B(real_A)
        synthetic_A = self.G_B2A(real_B)
        dA_guess_synthetic = self.D_A_static(synthetic_A)
        dB_guess_synthetic = self.D_B_static(synthetic_B)
        reconstructed_A = self.G_B2A(synthetic_B)
        reconstructed_B = self.G_A2B(synthetic_A)

        model_outputs = [reconstructed_A, reconstructed_B]
        compile_losses = [self.cycle_loss, self.cycle_loss,
                          self.lse, self.lse]
        compile_weights = [self.lambda_1, self.lambda_2,
                           self.lambda_D, self.lambda_D]

        if self.use_multiscale_discriminator:
            for _ in range(2):
                compile_losses.append(self.lse)
                compile_weights.append(self.lambda_D)  # * 1e-3)  # Lower weight to regularize the model
            for i in range(2):
                model_outputs.append(dA_guess_synthetic[i])
                model_outputs.append(dB_guess_synthetic[i])
        else:
            model_outputs.append(dA_guess_synthetic)
            model_outputs.append(dB_guess_synthetic)

        if self.use_supervised_learning:
            model_outputs.append(synthetic_A)
            model_outputs.append(synthetic_B)
            compile_losses.append('MAE')
            compile_losses.append('MAE')
            compile_weights.append(self.supervised_weight)
            compile_weights.append(self.supervised_weight)

        self.G_model = Model(inputs=[real_A, real_B],
                             outputs=model_outputs,
                             name='G_model')

        self.G_model.compile(optimizer=self.opt_G,
                             loss=compile_losses,
                             loss_weights=compile_weights)
        # self.G_A2B.summary()

        # ======= Data ==========
        # Use 'None' to fetch all available images
        nr_A_train_imgs = None
        nr_B_train_imgs = None
        nr_A_test_imgs = None
        nr_B_test_imgs = None

        if self.use_data_generator:
            logging.info("Using data generator.")
        else:
            logging.info("Caching data in memory.")

        data = load_data.load_data(
            num_channels=self.channels,
            batch_size=self.batch_size,
            nr_A_train_imgs=nr_A_train_imgs,
            nr_B_train_imgs=nr_B_train_imgs,
            nr_A_test_imgs=nr_A_test_imgs,
            nr_B_test_imgs=nr_B_test_imgs,
            source_images=source_images,
            return_generator=self.use_data_generator
        )

        self.A_train = data["train_A_images"]
        self.B_train = data["train_B_images"]
        self.A_test = data["test_A_images"]
        self.B_test = data["test_B_images"]
        self.test_A_image_names = data["test_A_image_names"]
        self.test_B_image_names = data["test_B_image_names"]
        self.data_generator_train = data["train_AB_images_generator"]
        self.data_generator_test = data["test_AB_images_generator"]
        logging.info("Data loaded.")

        if not generate_synthetic_images:
            self.writeMetaDataToJSON()

        # ======= Avoid pre-allocating GPU memory ==========
        # TensorFlow wizardry
        tf_config = tf.compat.v1.ConfigProto()

        # Don't pre-allocate memory; allocate as-needed
        tf_config.gpu_options.allow_growth = True  # pylint: disable=no-member

        # Create a session with the above options specified.
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf_config))

        # ======= Initialize training ==========
        sys.stdout.flush()
        #plot_model(self.G_A2B, to_file='GA2B_expanded_model_new.png', show_shapes=True)
        if generate_synthetic_images:
            self.load_model_and_generate_synthetic_images()
        else:
            self.train(
                epochs=self.epochs,
                batch_size=self.batch_size,
                save_interval=self.save_interval
            )

#===============================================================================
# Architecture functions

    def ck(self, x, k, use_normalization, stride):
        x = Conv2D(filters=k, kernel_size=4, strides=stride, padding='same')(x)
        # Normalization is not done on the first discriminator layer
        if use_normalization:
            x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def c7Ak(self, x, k):
        x = Conv2D(filters=k, kernel_size=7, strides=1, padding='valid')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def dk(self, x, k):
        x = Conv2D(filters=k, kernel_size=3, strides=2, padding='same')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def Rk(self, x0):
        k = int(x0.shape[-1])
        # first layer
        x = ReflectionPadding2D((1, 1))(x0)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        # second layer
        x = ReflectionPadding2D((1, 1))(x)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        # merge
        x = add([x, x0])
        return x

    def uk(self, x, k):
        # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
        if self.use_resize_convolution:
            x = UpSampling2D(size=(2, 2))(x)  # Nearest neighbor upsampling
            x = ReflectionPadding2D((1, 1))(x)
            x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        else:
            x = Conv2DTranspose(filters=k, kernel_size=3, strides=2, padding='same')(x)  # this matches fractinoally stided with stride 1/2
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

#===============================================================================
# Models

    def modelMultiScaleDiscriminator(self, name=None):
        x1 = Input(shape=self.img_shape)
        x2 = AveragePooling2D(pool_size=(2, 2))(x1)
        #x4 = AveragePooling2D(pool_size=(2, 2))(x2)

        out_x1 = self.modelDiscriminator('D1')(x1)
        out_x2 = self.modelDiscriminator('D2')(x2)
        #out_x4 = self.modelDiscriminator('D4')(x4)

        return Model(inputs=x1, outputs=[out_x1, out_x2], name=name)

    def modelDiscriminator(self, name=None):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1 (#Instance normalization is not used for this layer)
        x = self.ck(input_img, 64, False, 2)
        # Layer 2
        x = self.ck(x, 128, True, 2)
        # Layer 3
        x = self.ck(x, 256, True, 2)
        # Layer 4
        x = self.ck(x, 512, True, 1)
        # Output layer
        if self.use_patchgan:
            x = Conv2D(filters=1, kernel_size=4, strides=1, padding='same')(x)
        else:
            x = Flatten()(x)
            x = Dense(1)(x)
        #x = Activation('sigmoid')(x) - No sigmoid to avoid near-fp32 machine epsilon discriminator cost
        return Model(inputs=input_img, outputs=x, name=name)

    def modelGenerator(self, name=None):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1
        x = ReflectionPadding2D((3, 3))(input_img)
        x = self.c7Ak(x, 32)
        # Layer 2
        x = self.dk(x, 64)
        # Layer 3
        x = self.dk(x, 128)

        if self.use_multiscale_discriminator:
            # Layer 3.5
            x = self.dk(x, 256)

        # Layer 4-12: Residual layer
        for _ in range(4, 13):
            x = self.Rk(x)

        if self.use_multiscale_discriminator:
            # Layer 12.5
            x = self.uk(x, 128)

        # Layer 13
        x = self.uk(x, 64)
        # Layer 14
        x = self.uk(x, 32)
        x = ReflectionPadding2D((3, 3))(x)
        x = Conv2D(self.channels, kernel_size=7, strides=1)(x)
        x = Activation('tanh')(x)  # They say they use Relu but really they do not
        return Model(inputs=input_img, outputs=x, name=name)

#===============================================================================
# Training
    def train(self, epochs, batch_size=1, save_interval=1):
        #tf.compat.v1.global_variables_initializer()
        def run_training_iteration(loop_index, epoch_iterations):
            # ======= Discriminator training ==========
            # Generate batch of synthetic images
            synthetic_images_B = self.G_A2B.predict(real_images_A)
            synthetic_images_A = self.G_B2A.predict(real_images_B)
            synthetic_images_A = synthetic_pool_A.query(synthetic_images_A)
            synthetic_images_B = synthetic_pool_B.query(synthetic_images_B)

            for _ in range(self.discriminator_iterations):
                DA_loss_real = self.D_A.train_on_batch(x=real_images_A, y=ones)
                DB_loss_real = self.D_B.train_on_batch(x=real_images_B, y=ones)
                DA_loss_synthetic = self.D_A.train_on_batch(x=synthetic_images_A, y=zeros)
                DB_loss_synthetic = self.D_B.train_on_batch(x=synthetic_images_B, y=zeros)
                if self.use_multiscale_discriminator:
                    DA_loss = sum(DA_loss_real) + sum(DA_loss_synthetic)
                    DB_loss = sum(DB_loss_real) + sum(DB_loss_synthetic)
                    print('DA_losses: ', np.add(DA_loss_real, DA_loss_synthetic))
                    print('DB_losses: ', np.add(DB_loss_real, DB_loss_synthetic))
                else:
                    DA_loss = DA_loss_real + DA_loss_synthetic
                    DB_loss = DB_loss_real + DB_loss_synthetic
                D_loss = DA_loss + DB_loss

                if self.discriminator_iterations > 1:
                    print('D_loss:', D_loss)
                    sys.stdout.flush()

            # ======= Generator training ==========
            target_data = [real_images_A, real_images_B]  # Compare reconstructed images to real images
            if self.use_multiscale_discriminator:
                for i in range(2):
                    target_data.append(ones[i])
                    target_data.append(ones[i])
            else:
                target_data.append(ones)
                target_data.append(ones)

            if self.use_supervised_learning:
                target_data.append(real_images_A)
                target_data.append(real_images_B)

            for _ in range(self.generator_iterations):
                G_loss = self.G_model.train_on_batch(
                    x=[real_images_A, real_images_B], y=target_data)
                if self.generator_iterations > 1:
                    print('G_loss:', G_loss)
                    sys.stdout.flush()

            gA_d_loss_synthetic = G_loss[1]
            gB_d_loss_synthetic = G_loss[2]
            reconstruction_loss_A = G_loss[3]
            reconstruction_loss_B = G_loss[4]

            # Identity training
            if self.use_identity_learning and loop_index % self.identity_mapping_modulus == 0:
                G_A2B_identity_loss = self.G_A2B.train_on_batch(
                    x=real_images_B, y=real_images_B)
                G_B2A_identity_loss = self.G_B2A.train_on_batch(
                    x=real_images_A, y=real_images_A)
                print('G_A2B_identity_loss:', G_A2B_identity_loss)
                print('G_B2A_identity_loss:', G_B2A_identity_loss)

            # Update learning rates
            if self.use_linear_decay and epoch > self.decay_epoch:
                self.update_lr(self.D_A, decay_D)
                self.update_lr(self.D_B, decay_D)
                self.update_lr(self.G_model, decay_G)

            # Store training data
            DA_losses.append(DA_loss)
            DB_losses.append(DB_loss)
            gA_d_losses_synthetic.append(gA_d_loss_synthetic)
            gB_d_losses_synthetic.append(gB_d_loss_synthetic)
            gA_losses_reconstructed.append(reconstruction_loss_A)
            gB_losses_reconstructed.append(reconstruction_loss_B)

            GA_loss = gA_d_loss_synthetic + reconstruction_loss_A
            GB_loss = gB_d_loss_synthetic + reconstruction_loss_B
            D_losses.append(D_loss)
            GA_losses.append(GA_loss)
            GB_losses.append(GB_loss)
            G_losses.append(G_loss)
            reconstruction_loss = reconstruction_loss_A + reconstruction_loss_B
            reconstruction_losses.append(reconstruction_loss)

            print('\n')
            print('Epoch----------------', epoch, '/', epochs)
            print('Loop index----------------', loop_index + 1, '/', epoch_iterations)
            print('D_loss: ', D_loss)
            print('G_loss: ', G_loss[0])
            print('reconstruction_loss: ', reconstruction_loss)
            print('dA_loss:', DA_loss)
            print('DB_loss:', DB_loss)

            if loop_index % 20 == 0:
                # Save temporary images continously
                self.save_tmp_images(real_images_A, real_images_B, synthetic_images_A, synthetic_images_B)
                self.print_ETA(start_time, epoch, epoch_iterations, loop_index)

        # ======================================================================
        # Begin training
        # ======================================================================
        training_history = OrderedDict()

        DA_losses = []
        DB_losses = []
        gA_d_losses_synthetic = []
        gB_d_losses_synthetic = []
        gA_losses_reconstructed = []
        gB_losses_reconstructed = []

        GA_losses = []
        GB_losses = []
        reconstruction_losses = []
        D_losses = []
        G_losses = []

        # Image pools used to update the discriminators
        synthetic_pool_A = ImagePool(self.synthetic_pool_size)
        synthetic_pool_B = ImagePool(self.synthetic_pool_size)

        # self.saveImages('(init)')

        # labels
        if self.use_multiscale_discriminator:
            label_shape1 = (batch_size,) + self.D_A.output_shape[0][1:]
            label_shape2 = (batch_size,) + self.D_A.output_shape[1][1:]
            #label_shape4 = (batch_size,) + self.D_A.output_shape[2][1:]
            ones1 = np.ones(shape=label_shape1) * self.REAL_LABEL
            ones2 = np.ones(shape=label_shape2) * self.REAL_LABEL
            #ones4 = np.ones(shape=label_shape4) * self.REAL_LABEL
            ones = [ones1, ones2]  # , ones4]
            zeros1 = ones1 * 0
            zeros2 = ones2 * 0
            #zeros4 = ones4 * 0
            zeros = [zeros1, zeros2]  # , zeros4]
        else:
            label_shape = (batch_size,) + self.D_A.output_shape[1:]
            ones = np.ones(shape=label_shape) * self.REAL_LABEL
            zeros = ones * 0

        # Linear decay
        if self.use_linear_decay:
            decay_D, decay_G = self.get_lr_linear_decay_rate()

        # Start stopwatch for ETAs
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            if self.use_data_generator:
                loop_index = 1
                for batch in self.data_generator_train:
                    real_images_A = batch['real_images_A']
                    real_images_B = batch['real_images_B']
                    if len(real_images_A.shape) == 3:
                        real_images_A = real_images_A[:, :, :, np.newaxis]
                        real_images_B = real_images_B[:, :, :, np.newaxis]

                    # Run all training steps
                    run_training_iteration(loop_index, self.data_generator_train.__len__())

                    # Store models
                    if loop_index % 20000 == 0:
                        self.saveModel(self.D_A, loop_index)
                        self.saveModel(self.D_B, loop_index)
                        self.saveModel(self.G_A2B, loop_index)
                        self.saveModel(self.G_B2A, loop_index)

                    # Break if loop has ended
                    if loop_index >= self.data_generator_train.__len__():
                        break

                    loop_index += 1

            else:  # Train with all data in cache
                A_train = self.A_train
                B_train = self.B_train
                random_order_A = np.random.randint(len(A_train), size=len(A_train))
                random_order_B = np.random.randint(len(B_train), size=len(B_train))
                epoch_iterations = max(len(random_order_A), len(random_order_B))
                min_nr_imgs = min(len(random_order_A), len(random_order_B))

                # If we want supervised learning the same images form
                # the two domains are needed during each training iteration
                if self.use_supervised_learning:
                    random_order_B = random_order_A
                for loop_index in range(0, epoch_iterations, batch_size):
                    if loop_index + batch_size >= min_nr_imgs:
                        # If all images soon are used for one domain,
                        # randomly pick from this domain
                        if len(A_train) <= len(B_train):
                            indexes_A = np.random.randint(len(A_train), size=batch_size)

                            # if all images are used for the other domain
                            if loop_index + batch_size >= epoch_iterations:
                                indexes_B = random_order_B[
                                    epoch_iterations-batch_size:epoch_iterations
                                ]
                            else: # if not used, continue iterating...
                                indexes_B = random_order_B[loop_index:
                                                           loop_index + batch_size]

                        else: # if len(B_train) <= len(A_train)
                            indexes_B = np.random.randint(len(B_train), size=batch_size)
                            # if all images are used for the other domain
                            if loop_index + batch_size >= epoch_iterations:
                                indexes_A = random_order_A[
                                    epoch_iterations-batch_size:epoch_iterations
                                ]
                            else: # if not used, continue iterating...
                                indexes_A = random_order_A[
                                    loop_index:loop_index + batch_size
                                ]
                    else:
                        indexes_A = random_order_A[loop_index:
                                                   loop_index + batch_size]
                        indexes_B = random_order_B[loop_index:
                                                   loop_index + batch_size]

                    sys.stdout.flush()
                    real_images_A = A_train[indexes_A]
                    real_images_B = B_train[indexes_B]

                    # Run all training steps
                    run_training_iteration(loop_index, epoch_iterations)

            #================== within epoch loop end ==========================

            if epoch % save_interval == 0:
                print('\n', '\n', '-------------------------Saving images for epoch', epoch, '-------------------------', '\n', '\n')
                self.saveImages(epoch, real_images_A, real_images_B)

            if epoch % 20 == 0:
                # self.saveModel(self.G_model)
                self.saveModel(self.D_A, epoch)
                self.saveModel(self.D_B, epoch)
                self.saveModel(self.G_A2B, epoch)
                self.saveModel(self.G_B2A, epoch)

            training_history = {
                'DA_losses': DA_losses,
                'DB_losses': DB_losses,
                'gA_d_losses_synthetic': gA_d_losses_synthetic,
                'gB_d_losses_synthetic': gB_d_losses_synthetic,
                'gA_losses_reconstructed': gA_losses_reconstructed,
                'gB_losses_reconstructed': gB_losses_reconstructed,
                'D_losses': D_losses,
                'G_losses': G_losses,
                'reconstruction_losses': reconstruction_losses}
            self.writeLossDataToFile(training_history)

            # Flush out prints each loop iteration
            sys.stdout.flush()

#===============================================================================
# Help functions

    def lse(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.math.squared_difference(y_pred, y_true))
        return loss

    def cycle_loss(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.abs(y_pred - y_true))
        return loss

    def truncateAndSave(self, real_, real, synthetic, reconstructed, path_name):
        if len(real.shape) > 3:
            real = real[0]
            synthetic = synthetic[0]
            reconstructed = reconstructed[0]

        # Append and save
        if real_ is not None:
            if len(real_.shape) > 4:
                real_ = real_[0]
            image = np.hstack((real_[0], real, synthetic, reconstructed))
        else:
            image = np.hstack((real, synthetic, reconstructed))

        if self.channels == 1:
            image = image[:, :, 0]

        PIL.Image.fromarray(
            ((image + 1) * 127.5).astype('uint8')
        ).save(path_name)

    def saveImages(self, epoch, real_image_A, real_image_B, num_saved_images=1):
        real_image_Ab = None
        real_image_Ba = None
        for i in range(num_saved_images + 1):
            if i == num_saved_images:
                train_or_test = 'test'
                real_image_A = self.A_test[0]
                real_image_B = self.B_test[0]
                real_image_A = np.expand_dims(real_image_A, axis=0)
                real_image_B = np.expand_dims(real_image_B, axis=0)
            else:
                train_or_test = 'train'
                if len(real_image_A.shape) < 4:
                    real_image_A = np.expand_dims(real_image_A, axis=0)
                    real_image_B = np.expand_dims(real_image_B, axis=0)

            synthetic_image_B = self.G_A2B.predict(real_image_A)
            synthetic_image_A = self.G_B2A.predict(real_image_B)
            reconstructed_image_A = self.G_B2A.predict(synthetic_image_B)
            reconstructed_image_B = self.G_A2B.predict(synthetic_image_A)

            self.truncateAndSave(
                real_image_Ab,
                real_image_A,
                synthetic_image_B,
                reconstructed_image_A,
                self.result_paths.__dict__[
                    f'output_history_samples_{train_or_test}_A'
                ] / f'epoch{epoch:04d}_sample{i}.png'
            )
            self.truncateAndSave(
                real_image_Ba,
                real_image_B,
                synthetic_image_A,
                reconstructed_image_B,
                self.result_paths.__dict__[
                    f'output_history_samples_{train_or_test}_B'
                ] / f'epoch{epoch:04d}_sample{i}.png'
            )

    def save_tmp_images(self, real_image_A, real_image_B, synthetic_image_A, synthetic_image_B):
        try:
            reconstructed_image_A = self.G_B2A.predict(synthetic_image_B)
            reconstructed_image_B = self.G_A2B.predict(synthetic_image_A)

            real_images = np.vstack((real_image_A[0], real_image_B[0]))
            synthetic_images = np.vstack((synthetic_image_B[0], synthetic_image_A[0]))
            reconstructed_images = np.vstack((reconstructed_image_A[0], reconstructed_image_B[0]))

            self.truncateAndSave(
                None,
                real_images,
                synthetic_images,
                reconstructed_images,
                self.result_paths.output_history_samples / 'tmp.png'
            )
        except Exception:  # pylint: disable=broad-except
            pass

    def get_lr_linear_decay_rate(self):
        # Calculate decay rates
        if self.use_data_generator:
            max_nr_images = len(self.data_generator_train)
        else:
            max_nr_images = max(len(self.A_train), len(self.B_train))

        updates_per_epoch_D = 2 * max_nr_images + self.discriminator_iterations - 1
        updates_per_epoch_G = max_nr_images + self.generator_iterations - 1
        if self.use_identity_learning:
            updates_per_epoch_G *= (1 + 1 / self.identity_mapping_modulus)
        denominator_D = (self.epochs - self.decay_epoch) * updates_per_epoch_D
        denominator_G = (self.epochs - self.decay_epoch) * updates_per_epoch_G
        decay_D = self.learning_rate_D / denominator_D
        decay_G = self.learning_rate_G / denominator_G

        return decay_D, decay_G

    def update_lr(self, model, decay):
        new_lr = K.get_value(model.optimizer.lr) - decay
        if new_lr < 0:
            new_lr = 0
        # print(K.get_value(model.optimizer.lr))
        K.set_value(model.optimizer.lr, new_lr)

    def print_ETA(self, start_time, epoch, epoch_iterations, loop_index):
        passed_time = time.time() - start_time

        iterations_so_far = ((epoch - 1) * epoch_iterations + loop_index) / self.batch_size
        iterations_total = self.epochs * epoch_iterations / self.batch_size
        iterations_left = iterations_total - iterations_so_far
        eta = round(passed_time / (iterations_so_far + 1e-5) * iterations_left)

        passed_time_string = str(datetime.timedelta(seconds=round(passed_time)))
        eta_string = str(datetime.timedelta(seconds=eta))
        print('Time passed', passed_time_string, ': ETA in', eta_string)


#===============================================================================
# Save and load

    def saveModel(self, model, epoch):
        model_path_m = self.result_paths.saved_models / f'{model.name}_model.json'  # Architecture constant accross epochs.
        model_path_w = self.result_paths.saved_models / f'{model.name}_weights_epoch_{epoch:04d}.hdf5'
        model.save_weights(str(model_path_w))
        json_string = model.to_json()
        with model_path_m.open('w') as outfile:
            json.dump(json_string, outfile, indent=4)
        print(f"{model.name} has been saved in {self.result_paths.saved_models}.")

    def writeLossDataToFile(self, history):
        keys = sorted(history.keys())
        with (self.result_paths.base / 'loss_output.csv').open('w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(keys)
            writer.writerows(zip(*[history[key] for key in keys]))

    def writeMetaDataToJSON(self):
        data = {
            'img shape: height, width, channels': self.img_shape,
            'batch size': self.batch_size,
            'save interval': self.save_interval,
            'normalization function': str(self.normalization),
            'lambda_1': self.lambda_1,
            'lambda_2': self.lambda_2,
            'lambda_d': self.lambda_D,
            'learning_rate_D': self.learning_rate_D,
            'learning rate G': self.learning_rate_G,
            'epochs': self.epochs,
            'use linear decay on learning rates': self.use_linear_decay,
            'use multiscale discriminator': self.use_multiscale_discriminator,
            'epoch where learning rate linear decay is initialized (if use_linear_decay)': self.decay_epoch,
            'generator iterations': self.generator_iterations,
            'discriminator iterations': self.discriminator_iterations,
            'use patchGan in discriminator': self.use_patchgan,
            'beta 1': self.beta_1,
            'beta 2': self.beta_2,
            'REAL_LABEL': self.REAL_LABEL,
            'source_images': str(self.source_images),
            'num_train_examples': len(
                self.data_generator_train
                if self.data_generator_train
                else self.A_train
            )
        }

        with (self.result_paths.base / 'meta_data.json').open('w') as outfile:
            json.dump(data, outfile, sort_keys=True, indent=4)

    def load_weights_for_model(self, model):
        path_to_weights = sorted(
            self.result_paths.saved_models.glob(f'{model.name}_weights_epoch_*.hdf5')
        )[-1]  # Load last available weights.
        logging.info("Loading weights from %s.", path_to_weights)
        model.load_weights(str(path_to_weights))

    def load_model_and_generate_synthetic_images(self):
        self.load_weights_for_model(self.G_A2B)
        self.load_weights_for_model(self.G_B2A)

        def save_image(image, name, domain):
            if self.channels == 1:
                image = image[:, :, 0]
            PIL.Image.fromarray(
                ((image + 1) * 127.5).astype('uint8')
            ).save(
                self.result_paths.__dict__[
                    f'generated_synthetic_images_{domain}'
                ] / name
            )

        logging.info("Running prediction on test images.")

        if self.use_data_generator:
            loop_index = 1
            for batch in tqdm.tqdm(self.data_generator_test):
                real_images_A = batch['real_images_A']
                real_images_B = batch['real_images_B']
                real_image_paths_A = batch['real_image_paths_A']
                real_image_paths_B = batch['real_image_paths_B']
                synthetic_images_B = self.G_A2B.predict(real_images_A)
                synthetic_images_A = self.G_B2A.predict(real_images_B)
                for i in range(len(synthetic_images_A)):
                    name_A = real_image_paths_B[i].stem + '_synthetic.png'
                    synthetic_image_A = synthetic_images_A[i]
                    save_image(synthetic_image_A, name_A, 'A')
                for i in range(len(synthetic_images_B)):
                    name_B = real_image_paths_A[i].stem + '_synthetic.png'
                    synthetic_image_B = synthetic_images_B[i]
                    save_image(synthetic_image_B, name_B, 'B')
                if loop_index >= self.data_generator_test.__len__():
                    break
                loop_index += 1
            logging.info(
                f"{len(self.data_generator_test) * self.batch_size} synthetic images have been generated and "
                f"placed in {self.result_paths.generated_synthetic_images_A}."
            )
            logging.info(
                f"{len(self.data_generator_test) * self.batch_size} synthetic images have been generated and "
                f"placed in {self.result_paths.generated_synthetic_images_B}."
            )
        else:
            synthetic_images_B = self.G_A2B.predict(self.A_test)
            synthetic_images_A = self.G_B2A.predict(self.B_test)
            for i in range(len(synthetic_images_A)):
                name_A = self.test_B_image_names[i].stem + '_synthetic.png'
                synthetic_image_A = synthetic_images_A[i]
                save_image(synthetic_image_A, name_A, 'A')
            for i in range(len(synthetic_images_B)):
                name_B = self.test_A_image_names[i].stem + '_synthetic.png'
                synthetic_image_B = synthetic_images_B[i]
                save_image(synthetic_image_B, name_B, 'B')
            logging.info(
                f"{len(self.A_test)} synthetic images have been generated and "
                f"placed in {self.result_paths.generated_synthetic_images_A}."
            )
            logging.info(
                f"{len(self.B_test)} synthetic images have been generated and "
                f"placed in {self.result_paths.generated_synthetic_images_B}."
            )


# reflection padding taken from
# https://github.com/fastai/courses/blob/master/deeplearning2/neural-style.ipynb
class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            input_shape[1] + 2 * self.padding[0],
            input_shape[2] + 2 * self.padding[1],
            input_shape[3]
        )

    def call(self, inputs, **kwargs):
        w_pad, h_pad = self.padding
        return tf.pad(
            inputs,
            [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]],
            'REFLECT'
        )

    def get_config(self):  # pylint: disable=useless-super-delegation
        # TODO: Add padding arguments to returned config.
        return super().get_config()


class ImagePool():
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

            else:  # 50% chance that we replace an old synthetic image
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


def get_arguments():
    parser = argparse.ArgumentParser(
        usage=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config-path',
        help=f"JSON config path, relative to {config.STATIC_PATHS.configs}."
    )
    parser.add_argument(
        '--model-key', default=None,
        help=(
            f"Load model from this key relative to {config.STATIC_PATHS.results}. "
            "If supplied, model is loaded from this key rather than trained. "
            "The chronologically last model checkpoint is loaded."
        )
    )
    parser.add_argument(
        '--generate-synthetic-images',
        dest='generate_synthetic_images', action='store_true',
        help=(
            "Do not train, but load model from --model-key and generate "
            "synthetic images for the entire test set."
        )
    )
    parser.set_defaults(generate_synthetic_images=False)
    parser.add_argument(
        '--verbose-tensorflow', dest='verbose_tensorflow', action='store_true',
        help="Do not suppress TensorFlow logging output."
    )
    parser.add_argument(
        '--silent-tensorflow', dest='verbose_tensorflow', action='store_false',
        help="Suppress TensorFlow logging output."
    )
    parser.set_defaults(verbose_tensorflow=False)

    return parser.parse_args()


def get_config(config_path):
    return config.train_config_from_json(config_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arguments = get_arguments()
    logging.info("Running with the following arguments.")
    logging.info(arguments)
    config_ = get_config(arguments.config_path)
    logging.info("Running with the following config.")
    logging.info(config_)
    os.nice(3)
    # ^ Increase nice level to avoid ssh-lockout in case the process
    # exhausts system resources.
    if arguments.verbose_tensorflow:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        tf.compat.v1.logging.set_verbosity(
            tf.compat.v1.logging.DEBUG
        )
    else:
        # Disable TensorFlow spam.
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.compat.v1.logging.set_verbosity(
            tf.compat.v1.logging.ERROR
        )

    GAN = CycleGAN(
        image_shape=config_.image_shape,
        use_data_generator=config_.use_data_generator,
        source_images=config_.source_images,
        batch_size=config_.batch_size,
        model_key=arguments.model_key,
        generate_synthetic_images=arguments.generate_synthetic_images
    )
