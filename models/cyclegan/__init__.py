import csv
import json
import logging
import tqdm

import PIL.Image
import numpy as np
import tensorflow as tf

import config
import models.cyclegan.components as components
import models.io
import util


class CycleGAN():
    def __init__(
        self,
        config_,  # pylint: disable=redefined-outer-name
        generate_synthetic_images=False,
        model_key=None
    ):
        self.config = config_

        np.random.seed(4242)
        tf.random.set_seed(4242)

        self.prepare_optimizers()
        self.prepare_discriminators()
        self.prepare_generators()
        self.prepare_full_model()
        self.prepare_data()

        #tf.keras.utils.plot_model(self.G_A2B, to_file='GA2B_expanded_model_new.png', show_shapes=True)
        if generate_synthetic_images:
            self.result_paths = config.construct_result_paths(
                model_key=model_key
            )
            self.load_model_and_generate_synthetic_images()
        else:
            self.result_paths = config.construct_result_paths(
                create_dirs=True
            )
            models.io.save_metadata(
                data={
                    **self.config.__dict__,
                    **{
                        'num_train_examples': len(
                            self.data['train_AB_images_generator']
                            if self.data['train_AB_images_generator']
                            else self.data['train_A_images']
                        )
                    }
                },
                result_paths_base=self.result_paths.base
            )
            self.train()

    def prepare_optimizers(self):
        self.optimizer_D = tf.keras.optimizers.Adam(
            self.config.learning_rate_D,
            self.config.adam_beta_1,
            self.config.adam_beta_2
        )
        self.optimizer_G = tf.keras.optimizers.Adam(
            self.config.learning_rate_G,
            self.config.adam_beta_1,
            self.config.adam_beta_2
        )

    def prepare_discriminators(self):
        discriminator_args = dict(
            image_shape=self.config.image_shape,
            normalization=self.config.normalization,
            use_patchgan_discriminator=self.config.use_patchgan_discriminator
        )
        if self.config.use_multiscale_discriminator:
            self.D_A = components.multi_scale_discriminator(
                **discriminator_args, name='D_A'
            )
            self.D_B = components.multi_scale_discriminator(
                **discriminator_args, name='D_B'
            )
            loss_weights_D = [0.5, 0.5]
            # ^ 0.5 since we train on real and synthetic images.
        else:
            D_A = components.discriminator(
                **discriminator_args, name='D_A'
            )
            D_B = components.discriminator(
                **discriminator_args, name='D_B'
            )
            loss_weights_D = [0.5]
            # ^ 0.5 since we train on real and synthetic images.
        # D_A.summary()

        # Double the discriminators because discriminator weights
        # are not updated during generator training.
        # TODO: Find a more economic solution.
        image_A = tf.keras.Input(shape=self.config.image_shape)
        image_B = tf.keras.Input(shape=self.config.image_shape)
        guess_A = D_A(image_A)
        guess_B = D_B(image_B)
        self.D_A = tf.keras.Model(inputs=image_A, outputs=guess_A, name='D_A')
        self.D_B = tf.keras.Model(inputs=image_B, outputs=guess_B, name='D_B')
        self.D_A_static = tf.keras.Model(inputs=image_A, outputs=guess_A, name='D_A_static')
        self.D_B_static = tf.keras.Model(inputs=image_B, outputs=guess_B, name='D_B_static')
        self.D_A_static.trainable = False
        self.D_B_static.trainable = False

        # self.D_A.summary()
        # self.D_B.summary()
        self.D_A.compile(
            optimizer=self.optimizer_D,
            loss=util.tensorflow_mse,
            loss_weights=loss_weights_D
        )
        self.D_B.compile(
            optimizer=self.optimizer_D,
            loss=util.tensorflow_mse,
            loss_weights=loss_weights_D
        )

    def prepare_generators(self):
        generator_args = dict(
            image_shape=self.config.image_shape,
            normalization=self.config.normalization,
            use_multiscale_discriminator=self.config.use_multiscale_discriminator,
            use_resize_convolution=self.config.use_resize_convolution
        )
        self.G_A2B = components.generator(
            **generator_args, name='G_A2B'
        )
        self.G_B2A = components.generator(
            **generator_args, name='G_B2A'
        )
        # self.G_A2B.summary()
        if self.config.use_identity_learning:
            self.G_A2B.compile(optimizer=self.optimizer_G, loss='MAE')
            self.G_B2A.compile(optimizer=self.optimizer_G, loss='MAE')

    def prepare_full_model(self):
        real_A = tf.keras.Input(shape=self.config.image_shape, name='real_A')
        real_B = tf.keras.Input(shape=self.config.image_shape, name='real_B')
        synthetic_B = self.G_A2B(real_A)
        synthetic_A = self.G_B2A(real_B)
        D_A_guess_synthetic = self.D_A_static(synthetic_A)
        D_B_guess_synthetic = self.D_B_static(synthetic_B)
        reconstructed_A = self.G_B2A(synthetic_B)
        reconstructed_B = self.G_A2B(synthetic_A)

        model_outputs = [reconstructed_A, reconstructed_B]
        compile_losses = [util.tensorflow_mae, util.tensorflow_mae,
                          util.tensorflow_mse, util.tensorflow_mse]
        compile_weights = [self.config.lambda_1, self.config.lambda_2,
                           self.config.lambda_D, self.config.lambda_D]

        if self.config.use_multiscale_discriminator:
            for _ in range(2):
                compile_losses.append(util.tensorflow_mse)
                compile_weights.append(self.config.lambda_D)
            for i in range(2):
                model_outputs.append(D_A_guess_synthetic[i])
                model_outputs.append(D_B_guess_synthetic[i])
        else:
            model_outputs.append(D_A_guess_synthetic)
            model_outputs.append(D_B_guess_synthetic)

        if self.config.use_supervised_learning:
            model_outputs.append(synthetic_A)
            model_outputs.append(synthetic_B)
            compile_losses.append('MAE')
            compile_losses.append('MAE')
            compile_weights.append(self.config.supervised_learning_weight)
            compile_weights.append(self.config.supervised_learning_weight)

        self.G = tf.keras.Model(
            inputs=[real_A, real_B],
            outputs=model_outputs,
            name='G'
        )

        self.G.compile(optimizer=self.optimizer_G,
                             loss=compile_losses,
                             loss_weights=compile_weights)
        # self.G.summary()

    def prepare_data(self):
        if self.config.use_data_generator:
            logging.info("Using data generator.")
        else:
            logging.info("Caching data in memory.")

        self.data = models.io.load_data(
            num_channels=self.config.image_shape[-1],
            batch_size=self.config.batch_size,
            num_train_A_images=self.config.num_train_A_images,
            num_train_B_images=self.config.num_train_B_images,
            num_test_A_images=self.config.num_test_A_images,
            num_test_B_images=self.config.num_test_B_images,
            source_images=self.config.source_images,
            return_generator=self.config.use_data_generator
        )

        if self.config.use_data_generator:
            self.batches_per_epoch = len(self.data['train_AB_images_generator'])
            self.min_num_images = self.batches_per_epoch * self.config.batch_size
            self.max_num_images = self.min_num_images
        else:
            self.max_num_images = max(
                len(self.data['train_A_images']),
                len(self.data['train_B_images'])
            )
            self.min_num_images = min(
                len(self.data['train_A_images']),
                len(self.data['train_B_images'])
            )
            self.batches_per_epoch = (
                self.max_num_images // self.config.batch_size + 1
                if self.max_num_images % self.config.batch_size > 0
                else self.max_num_images // self.config.batch_size
            )

        # Image pools used to update the discriminators
        self.synthetic_pool_A = \
            components.ImagePool(self.config.synthetic_pool_size)
        self.synthetic_pool_B = \
            components.ImagePool(self.config.synthetic_pool_size)

        # labels
        if self.config.use_multiscale_discriminator:
            label_shape1 = (self.config.batch_size,) + self.D_A.output_shape[0][1:]
            label_shape2 = (self.config.batch_size,) + self.D_A.output_shape[1][1:]
            ones1 = np.ones(shape=label_shape1) * self.config.real_discriminator_label
            ones2 = np.ones(shape=label_shape2) * self.config.real_discriminator_label
            self.ones = [ones1, ones2]
            zeros1 = ones1 * 0
            zeros2 = ones2 * 0
            self.zeros = [zeros1, zeros2]
        else:
            label_shape = (self.config.batch_size,) + self.D_A.output_shape[1:]
            self.ones = np.ones(shape=label_shape) * self.config.real_discriminator_label
            self.zeros = self.ones * 0

        logging.info("Data prepared.")

    def train(self):
        self.D_A_losses = []
        self.D_B_losses = []
        self.G_A_D_losses_synthetic = []
        self.G_B_D_losses_synthetic = []
        self.G_A_losses_reconstructed = []
        self.G_B_losses_reconstructed = []

        self.G_A_losses = []
        self.G_B_losses = []
        self.reconstruction_losses = []
        self.D_losses = []
        self.G_losses = []

        if self.config.use_linear_lr_decay:
            self.calculate_learning_rate_decrements()

        self.progress = tqdm.tqdm(
            range(1, self.config.epochs + 1),
            unit='epoch'
        )
        for epoch in self.progress:
            self.epoch = epoch
            if self.config.use_data_generator:
                self.run_generator_training_epoch()
            else:
                self.run_ram_training_epoch()
            self.progress.set_postfix(**self.mean_epoch_losses())
            self.maybe_save_results()

    def run_generator_training_epoch(self):
        self.batch_index = 1
        self.sample_index = 0
        for batch in tqdm.tqdm(
            self.data['train_AB_images_generator'],
            unit='batch'
        ):
            self.real_images_batch_A = batch['self.real_images_batch_A']
            self.real_images_batch_B = batch['self.real_images_batch_B']
            self.run_batch_training_iteration()
            if self.batch_index >= self.batches_per_epoch:
                break
            self.sample_index = self.batch_index * self.config.batch_size
            self.batch_index += 1

    def run_ram_training_epoch(self):
        A_train = self.data['train_A_images']
        B_train = self.data['train_B_images']
        random_order_A = np.random.randint(len(A_train), size=len(A_train))
        random_order_B = np.random.randint(len(B_train), size=len(B_train))
        # If we want supervised learning the same images from
        # the two domains are needed during each training iteration
        if self.config.use_supervised_learning:
            random_order_B = random_order_A
        for sample_index in tqdm.tqdm(
            range(
                0,
                self.batches_per_epoch,
                self.config.batch_size
            ),
            unit='batch'
        ):
            self.sample_index = sample_index
            self.batch_index = self.sample_index // self.config.batch_size + 1
            if self.sample_index + self.config.batch_size >= self.min_num_images:
                # If all images soon are used for one domain,
                # randomly pick from this domain.
                if len(A_train) <= len(B_train):
                    indexes_A = np.random.randint(len(A_train), size=self.config.batch_size)
                    # if all images are used for the other domain
                    if self.sample_index + self.config.batch_size >= self.min_num_images:
                        indexes_B = random_order_B[
                            self.min_num_images-self.config.batch_size:self.min_num_images
                        ]
                    else: # if not used, continue iterating...
                        indexes_B = random_order_B[
                            self.sample_index:self.sample_index + self.config.batch_size
                        ]
                else: # if len(B_train) < len(A_train)
                    indexes_B = np.random.randint(
                        len(B_train), size=self.config.batch_size
                    )
                    # if all images are used for the other domain
                    if self.sample_index + self.config.batch_size >= self.min_num_images:
                        indexes_A = random_order_A[
                            self.min_num_images-self.config.batch_size:self.min_num_images
                        ]
                    else: # if not used, continue iterating...
                        indexes_A = random_order_A[
                            self.sample_index:self.sample_index + self.config.batch_size
                        ]
            else:
                indexes_A = random_order_A[
                    self.sample_index:self.sample_index + self.config.batch_size
                ]
                indexes_B = random_order_B[
                    self.sample_index:self.sample_index + self.config.batch_size
                ]
            self.real_images_batch_A = A_train[indexes_A]
            self.real_images_batch_B = B_train[indexes_B]
            self.run_batch_training_iteration()

    def run_batch_training_iteration(self):
        # Discriminator training
        # Generate batch of synthetic images
        synthetic_images_B = self.G_A2B.predict(self.real_images_batch_A)
        synthetic_images_A = self.G_B2A.predict(self.real_images_batch_B)
        synthetic_images_A = self.synthetic_pool_A.query(synthetic_images_A)
        synthetic_images_B = self.synthetic_pool_B.query(synthetic_images_B)

        for _ in range(self.config.discriminator_iterations):
            D_A_loss_real = self.D_A.train_on_batch(x=self.real_images_batch_A, y=self.ones)
            D_B_loss_real = self.D_B.train_on_batch(x=self.real_images_batch_B, y=self.ones)
            D_A_loss_synthetic = self.D_A.train_on_batch(x=synthetic_images_A, y=self.zeros)
            D_B_loss_synthetic = self.D_B.train_on_batch(x=synthetic_images_B, y=self.zeros)
            if self.config.use_multiscale_discriminator:
                D_A_loss = sum(D_A_loss_real) + sum(D_A_loss_synthetic)
                D_B_loss = sum(D_B_loss_real) + sum(D_B_loss_synthetic)
                logging.info(f"D_A_losses: {D_A_loss_real + D_A_loss_synthetic}")
                logging.info(f"D_B_losses: {D_B_loss_real + D_B_loss_synthetic}")
            else:
                D_A_loss = D_A_loss_real + D_A_loss_synthetic
                D_B_loss = D_B_loss_real + D_B_loss_synthetic
            D_loss = D_A_loss + D_B_loss

            if self.config.discriminator_iterations > 1:
                print('D_loss:', D_loss)

        # Generator training
        # Compare reconstructed images to real images
        target_data = [self.real_images_batch_A, self.real_images_batch_B]
        if self.config.use_multiscale_discriminator:
            for i in range(2):
                target_data.append(self.ones[i])
                target_data.append(self.ones[i])
        else:
            target_data.append(self.ones)
            target_data.append(self.ones)

        if self.config.use_supervised_learning:
            target_data.append(self.real_images_batch_A)
            target_data.append(self.real_images_batch_B)

        for _ in range(self.config.generator_iterations):
            G_loss = self.G.train_on_batch(
                x=[self.real_images_batch_A, self.real_images_batch_B],
                y=target_data
            )
            if self.config.generator_iterations > 1:
                print('G_loss:', G_loss)

        G_A_D_loss_synthetic = G_loss[1]
        G_B_D_loss_synthetic = G_loss[2]
        reconstruction_loss_A = G_loss[3]
        reconstruction_loss_B = G_loss[4]

        # Identity training
        if self.config.use_identity_learning and self.batch_index % self.config.identity_learning_modulus == 0:
            G_A2B_identity_loss = self.G_A2B.train_on_batch(
                x=self.real_images_batch_B, y=self.real_images_batch_B)
            G_B2A_identity_loss = self.G_B2A.train_on_batch(
                x=self.real_images_batch_A, y=self.real_images_batch_A)
            print('G_A2B_identity_loss:', G_A2B_identity_loss)
            print('G_B2A_identity_loss:', G_B2A_identity_loss)

        # Update learning rates
        if self.config.use_linear_lr_decay and self.epoch > self.config.linear_lr_decay_epoch_start:
            util.update_learning_rate(self.D_A, self.decrement_D)
            util.update_learning_rate(self.D_B, self.decrement_D)
            util.update_learning_rate(self.G, self.decrement_G)

        # Store training data
        self.D_A_losses.append(D_A_loss)
        self.D_B_losses.append(D_B_loss)
        self.G_A_D_losses_synthetic.append(G_A_D_loss_synthetic)
        self.G_B_D_losses_synthetic.append(G_B_D_loss_synthetic)
        self.G_A_losses_reconstructed.append(reconstruction_loss_A)
        self.G_B_losses_reconstructed.append(reconstruction_loss_B)
        GA_loss = G_A_D_loss_synthetic + reconstruction_loss_A
        GB_loss = G_B_D_loss_synthetic + reconstruction_loss_B
        self.D_losses.append(D_loss)
        self.G_A_losses.append(GA_loss)
        self.G_B_losses.append(GB_loss)
        self.G_losses.append(G_loss)
        reconstruction_loss = reconstruction_loss_A + reconstruction_loss_B
        self.reconstruction_losses.append(reconstruction_loss)

        if self.sample_index % 20 == 0:
            # Save temporary images continously
            self.save_tmp_images(
                self.real_images_batch_A, self.real_images_batch_B,
                synthetic_images_A, synthetic_images_B
            )

    def mean_epoch_losses(self):
        if self.D_losses:
            D_loss_mean = np.mean(self.D_losses[-self.batches_per_epoch:])
            G_loss_mean = np.mean([
                x[0]
                for x in self.G_losses[-self.batches_per_epoch:]
            ])
            reconstruction_loss_mean = np.mean(
                self.reconstruction_losses[-self.batches_per_epoch:]
            )
            D_A_loss_mean = np.mean(self.D_A_losses[-self.batches_per_epoch:])
            D_B_loss_mean = np.mean(self.D_B_losses[-self.batches_per_epoch:])
            return {
                'D': D_loss_mean,
                'G': G_loss_mean,
                'r': reconstruction_loss_mean,
                'DA': D_A_loss_mean,
                'DB': D_B_loss_mean
            }

    def maybe_save_results(self):
        if self.epoch % self.config.save_interval_samples == 0:
            self.save_example_images()

        if self.epoch % self.config.save_interval_model == 0:
            for model in [self.D_A, self.D_B, self.G_A2B, self.G_B2A]:
                models.io.save_model(
                    model, self.epoch, self.result_paths.saved_models
                )
            self.progress.set_postfix(info="Models saved.")

        training_history = {
            'D_A_losses': self.D_A_losses,
            'D_B_losses': self.D_B_losses,
            'G_A_D_losses_synthetic': self.G_A_D_losses_synthetic,
            'G_B_D_losses_synthetic': self.G_B_D_losses_synthetic,
            'G_A_losses_reconstructed': self.G_A_losses_reconstructed,
            'G_B_losses_reconstructed': self.G_B_losses_reconstructed,
            'D_losses': self.D_losses,
            'G_losses': self.G_losses,
            'reconstruction_losses': self.reconstruction_losses
        }
        models.io.save_losses(training_history, self.result_paths.base)

    def save_example_images(self, num_saved_images=1):
        for i in range(num_saved_images + 1):
            if i == num_saved_images:
                train_or_test = 'test'
                real_image_A = self.data['test_A_images'][0]
                real_image_B = self.data['test_B_images'][0]
            else:
                train_or_test = 'train'
                real_image_A = self.real_images_batch_A[i]
                real_image_B = self.real_images_batch_B[i]
            synthetic_image_B = self.G_A2B.predict(
                np.expand_dims(real_image_A, axis=0)
            )
            synthetic_image_A = self.G_B2A.predict(
                np.expand_dims(real_image_B, axis=0)
            )
            reconstructed_image_A = self.G_B2A.predict(synthetic_image_B)
            reconstructed_image_B = self.G_A2B.predict(synthetic_image_A)

            self.truncate_image_and_save(
                real_image_A,
                synthetic_image_B[0],
                reconstructed_image_A[0],
                self.result_paths.__dict__[
                    f'output_history_samples_{train_or_test}_A'
                ] / f'epoch{self.epoch:04d}_sample{i}.png'
            )
            self.truncate_image_and_save(
                real_image_B,
                synthetic_image_A[0],
                reconstructed_image_B[0],
                self.result_paths.__dict__[
                    f'output_history_samples_{train_or_test}_B'
                ] / f'epoch{self.epoch:04d}_sample{i}.png'
            )

    def save_tmp_images(self, real_image_A, real_image_B, synthetic_image_A, synthetic_image_B):
        try:
            reconstructed_image_A = self.G_B2A.predict(synthetic_image_B)
            reconstructed_image_B = self.G_A2B.predict(synthetic_image_A)

            real_images = np.vstack((real_image_A[0], real_image_B[0]))
            synthetic_images = np.vstack((synthetic_image_B[0], synthetic_image_A[0]))
            reconstructed_images = np.vstack((reconstructed_image_A[0], reconstructed_image_B[0]))

            self.truncate_image_and_save(
                real_images,
                synthetic_images,
                reconstructed_images,
                self.result_paths.output_history_samples / 'tmp.png'
            )
        except Exception:  # pylint: disable=broad-except
            pass

    def calculate_learning_rate_decrements(self):
        updates_per_epoch_D = 2 * self.max_num_images + self.config.discriminator_iterations - 1
        updates_per_epoch_G = self.max_num_images + self.config.generator_iterations - 1
        if self.config.use_identity_learning:
            updates_per_epoch_G *= (1 + 1 / self.config.identity_learning_modulus)
        denominator_D = (self.config.epochs - self.config.linear_lr_decay_epoch_start) * updates_per_epoch_D
        denominator_G = (self.config.epochs - self.config.linear_lr_decay_epoch_start) * updates_per_epoch_G
        self.decrement_D = self.config.learning_rate_D / denominator_D
        self.decrement_G = self.config.learning_rate_G / denominator_G

    def truncate_image_and_save(self, real, synthetic, reconstructed, path_name):
        image = np.hstack((real, synthetic, reconstructed))
        if self.config.image_shape[-1] == 1:
            image = image[:, :, 0]
        PIL.Image.fromarray(
            ((image + 1) * 127.5).astype('uint8')
        ).save(path_name)

    def load_model_and_generate_synthetic_images(self):
        models.io.load_weights_for_model(self.G_A2B, self.result_paths.saved_models)
        models.io.load_weights_for_model(self.G_B2A, self.result_paths.saved_models)

        def save_image(image, name, domain):
            if self.config.image_shape[-1] == 1:
                image = image[:, :, 0]
            PIL.Image.fromarray(
                ((image + 1) * 127.5).astype('uint8')
            ).save(
                self.result_paths.__dict__[
                    f'generated_synthetic_images_{domain}'
                ] / name
            )

        logging.info("Running prediction on test images.")

        if self.config.use_data_generator:
            self.batch_index = 1
            self.sample_index = 1
            for batch in tqdm.tqdm(
                self.data['test_AB_images_generator'],
                unit='batch'
            ):
                real_images_batch_A = batch['real_images_A']
                real_images_batch_B = batch['real_images_B']
                real_image_paths_A = batch['real_image_paths_A']
                real_image_paths_B = batch['real_image_paths_B']
                synthetic_images_B = self.G_A2B.predict(real_images_batch_A)
                synthetic_images_A = self.G_B2A.predict(real_images_batch_B)
                for i in range(len(synthetic_images_A)):
                    name_A = real_image_paths_B[i].stem + '_synthetic.png'
                    synthetic_image_A = synthetic_images_A[i]
                    save_image(synthetic_image_A, name_A, 'A')
                for i in range(len(synthetic_images_B)):
                    name_B = real_image_paths_A[i].stem + '_synthetic.png'
                    synthetic_image_B = synthetic_images_B[i]
                    save_image(synthetic_image_B, name_B, 'B')
                if self.batch_index >= len(self.data['test_AB_images_generator']):
                    break
                self.sample_index = self.batch_index * self.config.batch_size
                self.batch_index += 1
            logging.info(
                f"{len(self.data['test_AB_images_generator']) * self.config.batch_size} "
                "synthetic images have been generated and "
                f"placed in {self.result_paths.generated_synthetic_images_A}."
            )
            logging.info(
                f"{len(self.data['test_AB_images_generator']) * self.config.batch_size} "
                "synthetic images have been generated and "
                f"placed in {self.result_paths.generated_synthetic_images_B}."
            )
        else:
            synthetic_images_B = self.G_A2B.predict(self.data['test_A_images'])
            synthetic_images_A = self.G_B2A.predict(self.data['test_B_images'])
            for i in range(len(synthetic_images_A)):
                name_A = self.data['test_B_image_names'][i].stem + '_synthetic.png'
                synthetic_image_A = synthetic_images_A[i]
                save_image(synthetic_image_A, name_A, 'A')
            for i in range(len(synthetic_images_B)):
                name_B = self.data['test_A_image_names'][i].stem + '_synthetic.png'
                synthetic_image_B = synthetic_images_B[i]
                save_image(synthetic_image_B, name_B, 'B')
            logging.info(
                f"{len(self.data['test_A_images'])} synthetic images have been generated and "
                f"placed in {self.result_paths.generated_synthetic_images_A}."
            )
            logging.info(
                f"{len(self.data['test_B_images'])} synthetic images have been generated and "
                f"placed in {self.result_paths.generated_synthetic_images_B}."
            )
