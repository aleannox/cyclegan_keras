import csv
import json
import logging
import random
import tqdm

import PIL.Image
import numpy as np
import tensorflow as tf

import config
import models.cyclegan.components as components
import models.io
import util


class CycleGAN():
    def __init__(self, model_config):
        self.config = model_config

        random.seed(4242)
        np.random.seed(4242)
        tf.random.set_seed(4242)

        self.prepare_optimizers()
        self.prepare_discriminators()
        self.prepare_single_generators()
        self.prepare_full_model()

        # Attributes which are optionally filled later.
        self.result_paths = None
        self.data = None
        self.losses = None
        self.epoch = None
        self.learning_rate_decrements = None
        self.epoch_progress_bar = None

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
            D_A_proto = components.multi_scale_discriminator(
                **discriminator_args, name='D_A'
            )
            D_B_proto = components.multi_scale_discriminator(
                **discriminator_args, name='D_B'
            )
            loss_weights_D = [0.5, 0.5]
            # ^ 0.5 since we train on real and synthetic images.
        else:
            D_A_proto = components.discriminator(
                **discriminator_args, name='D_A'
            )
            D_B_proto = components.discriminator(
                **discriminator_args, name='D_B'
            )
            loss_weights_D = [0.5]
            # ^ 0.5 since we train on real and synthetic images.

        # Double the discriminators because discriminator weights
        # are not updated during generator training.
        # TODO: Find a more economic solution.
        image_A = tf.keras.Input(shape=self.config.image_shape)
        image_B = tf.keras.Input(shape=self.config.image_shape)
        guess_A = D_A_proto(image_A)
        guess_B = D_B_proto(image_B)
        self.D_A = tf.keras.Model(inputs=image_A, outputs=guess_A, name='D_A')
        self.D_B = tf.keras.Model(inputs=image_B, outputs=guess_B, name='D_B')
        self.D_A_static = tf.keras.Model(inputs=image_A, outputs=guess_A, name='D_A_static')
        self.D_B_static = tf.keras.Model(inputs=image_B, outputs=guess_B, name='D_B_static')
        self.D_A_static.trainable = False
        self.D_B_static.trainable = False

        self.D_A.compile(
            optimizer=self.optimizer_D,
            loss=tf.keras.losses.mse,
            loss_weights=loss_weights_D
        )
        self.D_B.compile(
            optimizer=self.optimizer_D,
            loss=tf.keras.losses.mse,
            loss_weights=loss_weights_D
        )

    def prepare_single_generators(self):
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
        if self.config.use_identity_learning:
            self.G_A2B.compile(
                optimizer=self.optimizer_G, loss=tf.keras.losses.mae
            )
            self.G_B2A.compile(
                optimizer=self.optimizer_G, loss=tf.keras.losses.ma
            )

    def prepare_full_model(self):
        # Inputs
        real_A = tf.keras.Input(shape=self.config.image_shape, name='real_A')
        real_B = tf.keras.Input(shape=self.config.image_shape, name='real_B')

        # Output candidates, wrapped in identities for renaming.
        synthetic_B = util.identity_layer('synthetic_B')(self.G_A2B(real_A))
        synthetic_A = util.identity_layer('synthetic_A')(self.G_B2A(real_B))
        if self.config.use_multiscale_discriminator:
            D_A_guess_synthetic = []
            D_B_guess_synthetic = []
            for i in range(2):
                D_A_guess_synthetic.append(
                    util.identity_layer(f'D_A_guess_synthetic_{i}')(
                        self.D_A_static(synthetic_A)[i]
                    )
                )
                D_B_guess_synthetic.append(
                    util.identity_layer(f'D_B_guess_synthetic_{i}')(
                        self.D_B_static(synthetic_B)[i]
                    )
                )
        else:
            D_A_guess_synthetic = util.identity_layer('D_A_guess_synthetic')(
                self.D_A_static(synthetic_A)
            )
            D_B_guess_synthetic = util.identity_layer('D_B_guess_synthetic')(
                self.D_B_static(synthetic_B)
            )
        reconstructed_A = \
            util.identity_layer('reconstructed_A')(self.G_B2A(synthetic_B))
        reconstructed_B = \
            util.identity_layer('reconstructed_B')(self.G_A2B(synthetic_A))

        # Outputs, losses, weights.

        # Cyclic generators
        outputs = [
            reconstructed_A,  # A -> B -> A
            reconstructed_B,  # A -> B -> A
        ]
        losses = {
            'reconstructed_A': tf.keras.losses.mae,
            'reconstructed_B': tf.keras.losses.mae
        }
        loss_weights = {
            'reconstructed_A': self.config.lambda_ABA,
            'reconstructed_B': self.config.lambda_BAB
        }

        # Discriminators A, B
        if self.config.use_multiscale_discriminator:
            for i in range(2):
                outputs.append(D_A_guess_synthetic[i])
                outputs.append(D_B_guess_synthetic[i])
                losses[f'D_A_guess_synthetic_{i}'] = \
                    tf.keras.losses.mse
                losses[f'D_B_guess_synthetic_{i}'] = \
                    tf.keras.losses.mse
                loss_weights[f'D_A_guess_synthetic_{i}'] = self.config.lambda_D
                loss_weights[f'D_B_guess_synthetic_{i}'] = self.config.lambda_D
        else:
            outputs.append(D_A_guess_synthetic)
            outputs.append(D_B_guess_synthetic)
            losses['D_A_guess_synthetic'] = tf.keras.losses.mse
            losses['D_B_guess_synthetic'] = tf.keras.losses.mse
            loss_weights['D_A_guess_synthetic'] = self.config.lambda_D
            loss_weights['D_B_guess_synthetic'] = self.config.lambda_D

        # Supervised individual generators B -> A, A -> B
        if self.config.use_supervised_learning:
            outputs.append(synthetic_A)
            outputs.append(synthetic_B)
            losses['synthetic_A'] = tf.keras.losses.mae
            losses['synthetic_B'] = tf.keras.losses.mae
            loss_weights['synthetic_A'] = self.config.supervised_learning_weight
            loss_weights['synthetic_B'] = self.config.supervised_learning_weight

        self.G = tf.keras.Model(
            inputs=[real_A, real_B],
            outputs=outputs,
            name='G'
        )
        self.G.compile(
            optimizer=self.optimizer_G,
            loss=losses,
            loss_weights=loss_weights
        )

    def prepare_data(self):
        # Input data.
        if self.config.use_data_generator:
            logging.info("Using data generator.")
        else:
            logging.info("Caching data in RAM.")
        self.data = models.io.load_data(self.config)

        # Labels.
        if self.config.use_multiscale_discriminator:
            self.data['positive_discriminator_labels'] = []
            self.data['negative_discriminator_labels'] = []
            for i in range(2):
                label_shape = \
                    (self.config.batch_size,) + self.D_A.output_shape[i][1:]
                self.data['positive_discriminator_labels'].append(
                    np.ones(shape=label_shape) *
                    self.config.real_discriminator_label
                )
                self.data['negative_discriminator_labels'].append(
                    np.zeros(shape=label_shape)
                )
        else:
            label_shape = (self.config.batch_size,) + self.D_A.output_shape[1:]
            self.data['positive_discriminator_labels'] = (
                np.ones(shape=label_shape) *
                self.config.real_discriminator_label
            )
            self.data['negative_discriminator_labels'] = \
                np.zeros(shape=label_shape)

        # Image pools used to update the discriminators.
        self.data['synthetic_pool_A'] = \
            components.ImagePool(self.config.synthetic_pool_size)
        self.data['synthetic_pool_B'] = \
            components.ImagePool(self.config.synthetic_pool_size)

        logging.info("Data prepared.")

    def train(self):
        self.result_paths = config.construct_result_paths(
            create_dirs=True
        )
        models.io.save_metadata(
            data={
                **self.config.__dict__,
                **{
                    'num_train_images_max':
                    self.data['num_train_images_max']
                }
            },
            result_paths_base=self.result_paths.base
        )
        self.losses = {}
        self.losses['D_A'] = []
        self.losses['D_B'] = []
        self.losses['G_A_D_synthetic'] = []
        self.losses['G_B_D_synthetic'] = []
        self.losses['G_A_reconstructed'] = []
        self.losses['G_B_reconstructed'] = []

        self.losses['G_A'] = []
        self.losses['G_B'] = []
        self.losses['reconstruction'] = []
        self.losses['D'] = []
        self.losses['G'] = []

        if self.config.use_identity_learning:
            self.losses['G_A2B_identity'] = []
            self.losses['G_B2A_identity'] = []

        if self.config.use_linear_lr_decay:
            self.calculate_learning_rate_decrements()

        self.epoch_progress_bar = tqdm.tqdm(
            range(1, self.config.epochs + 1),
            unit='epoch'
        )
        for epoch in self.epoch_progress_bar:
            self.epoch = epoch
            if self.config.use_data_generator:
                self.run_data_generator_training_epoch()
            else:
                self.run_ram_training_epoch()
            self.handle_epoch_callbacks()

    def run_data_generator_training_epoch(self):
        batch_progress_bar = tqdm.tqdm(
            self.data['train_batch_generator'],
            unit='batch'
        )
        for batch_index, batch in enumerate(batch_progress_bar):
            self.run_batch_training_iteration(batch_index, batch)

    def run_ram_training_epoch(self):
        random_A_order = np.random.randint(
            len(self.data['train_A_images']),
            size=len(self.data['train_A_images'])
        )
        # If we want supervised learning, A and B must match.
        if self.config.use_supervised_learning:
            random_B_order = random_A_order
        else:
            random_B_order = np.random.randint(
                len(self.data['train_B_images']),
                size=len(self.data['train_B_images'])
            )
        batch_progress_bar = tqdm.trange(
            self.data['num_train_batches'], unit='batch'
        )
        for batch_index in batch_progress_bar:
            A_batch = models.io.get_batch(
                batch_index,
                self.config.batch_size,
                self.data['train_A_images'][random_A_order],
                len(self.data['train_B_images'])
            )
            B_batch = models.io.get_batch(
                batch_index,
                self.config.batch_size,
                self.data['train_B_images'][random_B_order],
                len(self.data['train_A_images'])
            )
            batch = {
                'A_images': A_batch,
                'B_images': B_batch
            }
            self.run_batch_training_iteration(batch_index, batch)

    def run_batch_training_iteration(self, batch_index, batch):
        real_A_images = batch['A_images']
        real_B_images = batch['B_images']

        D_A_loss, D_B_loss = self.run_batch_training_iteration_discriminator(
            batch_index, real_A_images, real_B_images
        )

        (
            G_loss,
            G_A_D_loss_synthetic, G_B_D_loss_synthetic,
            reconstruction_loss_A, reconstruction_loss_B
        ) = self.run_batch_training_iteration_generator(
            real_A_images, real_B_images
        )

        G_A2B_identity_loss, G_B2A_identity_loss = \
            self.run_batch_training_iteration_identity(
                batch_index, real_A_images, real_B_images
            )
        
        self.record_losses(
            D_A_loss, D_B_loss,
            G_loss,
            G_A_D_loss_synthetic, G_B_D_loss_synthetic,
            reconstruction_loss_A, reconstruction_loss_B,
            G_A2B_identity_loss, G_B2A_identity_loss
        )

        self.maybe_update_learning_rates()

    def run_batch_training_iteration_discriminator(
        self, batch_index, real_A_images, real_B_images
    ):
        synthetic_B_images = self.G_A2B.predict(real_A_images)
        synthetic_A_images = self.G_B2A.predict(real_B_images)
        synthetic_A_images = \
            self.data['synthetic_pool_A'].query(synthetic_A_images)
        synthetic_B_images = \
            self.data['synthetic_pool_B'].query(synthetic_B_images)

        if batch_index % self.config.save_interval_temporary_examples == 0:
            self.save_temporary_example_images(
                real_A_images, real_B_images,
                synthetic_A_images, synthetic_B_images
            )

        for _ in range(self.config.discriminator_iterations):
            D_A_loss_real = self.D_A.train_on_batch(
                x=real_A_images,
                y=self.data['positive_discriminator_labels']
            )
            D_B_loss_real = self.D_B.train_on_batch(
                x=real_B_images,
                y=self.data['positive_discriminator_labels']
            )
            D_A_loss_synthetic = self.D_A.train_on_batch(
                x=synthetic_A_images,
                y=self.data['negative_discriminator_labels']
            )
            D_B_loss_synthetic = self.D_B.train_on_batch(
                x=synthetic_B_images,
                y=self.data['negative_discriminator_labels']
            )
            if self.config.use_multiscale_discriminator:
                D_A_loss = sum(D_A_loss_real) + sum(D_A_loss_synthetic)
                D_B_loss = sum(D_B_loss_real) + sum(D_B_loss_synthetic)
            else:
                D_A_loss = D_A_loss_real + D_A_loss_synthetic
                D_B_loss = D_B_loss_real + D_B_loss_synthetic
        return D_A_loss, D_B_loss

    def run_batch_training_iteration_generator(
        self, real_A_images, real_B_images
    ):
        x = {
            'real_A': real_A_images,
            'real_B': real_B_images,
        }
        y = {
            'reconstructed_A': real_A_images,
            'reconstructed_B': real_B_images,
        }
        if self.config.use_multiscale_discriminator:
            for i in range(2):
                y['D_A_guess_synthetic_{i}'] = \
                    self.data['positive_discriminator_labels'][i]
                y['D_B_guess_synthetic_{i}'] = \
                    self.data['positive_discriminator_labels'][i]
        else:
            y['D_A_guess_synthetic'] = \
                self.data['positive_discriminator_labels']
            y['D_B_guess_synthetic'] = \
                self.data['positive_discriminator_labels']
        if self.config.use_supervised_learning:
            y['synthetic_A'] = real_A_images
            y['synthetic_B'] = real_B_images

        for _ in range(self.config.generator_iterations):
            G_loss = self.G.train_on_batch(x=x, y=y, return_dict=True)

        if self.config.use_multiscale_discriminator:
            G_A_D_loss_synthetic = []
            G_B_D_loss_synthetic = []
            for i in range(2):
                G_A_D_loss_synthetic.append(
                    G_loss['D_A_guess_synthetic_{i}_loss']
                )
                G_B_D_loss_synthetic.append(
                    G_loss['D_B_guess_synthetic_{i}_loss']
                )
        else:
            G_A_D_loss_synthetic = G_loss['D_A_guess_synthetic_loss']
            G_B_D_loss_synthetic = G_loss['D_B_guess_synthetic_loss']
        reconstruction_loss_A = G_loss['reconstructed_A_loss']
        reconstruction_loss_B = G_loss['reconstructed_B_loss']

        return (
            G_loss,
            G_A_D_loss_synthetic, G_B_D_loss_synthetic,
            reconstruction_loss_A, reconstruction_loss_B
        )

    def run_batch_training_iteration_identity(
        self, batch_index, real_A_images, real_B_images
    ):
        if (
            self.config.use_identity_learning and
            batch_index % self.config.identity_learning_modulus == 0
        ):
            G_A2B_identity_loss = self.G_A2B.train_on_batch(
                x=real_B_images, y=real_B_images
            )
            G_B2A_identity_loss = self.G_B2A.train_on_batch(
                x=real_A_images, y=real_A_images
            )
        else:
            G_A2B_identity_loss, G_B2A_identity_loss = None, None
        return G_A2B_identity_loss, G_B2A_identity_loss

    def record_losses(
        self,
        D_A_loss, D_B_loss,
        G_loss,
        G_A_D_loss_synthetic, G_B_D_loss_synthetic,
        reconstruction_loss_A, reconstruction_loss_B,
        G_A2B_identity_loss, G_B2A_identity_loss
    ):
        self.losses['D_A'].append(D_A_loss)
        self.losses['D_B'].append(D_B_loss)
        self.losses['G_A_D_synthetic'].append(G_A_D_loss_synthetic)
        self.losses['G_B_D_synthetic'].append(G_B_D_loss_synthetic)
        self.losses['G_A_reconstructed'].append(reconstruction_loss_A)
        self.losses['G_B_reconstructed'].append(reconstruction_loss_B)
        G_A_loss = G_A_D_loss_synthetic + reconstruction_loss_A
        G_B_loss = G_B_D_loss_synthetic + reconstruction_loss_B
        self.losses['D'].append(D_A_loss + D_B_loss)
        self.losses['G_A'].append(G_A_loss)
        self.losses['G_B'].append(G_B_loss)
        self.losses['G'].append(G_loss)
        reconstruction_loss = reconstruction_loss_A + reconstruction_loss_B
        self.losses['reconstruction'].append(reconstruction_loss)
        if self.config.use_identity_learning:
            self.losses['G_A2B_identity'].append(G_A2B_identity_loss)
            self.losses['G_B2A_identity'].append(G_B2A_identity_loss)

    def maybe_update_learning_rates(self):
        if (
            self.config.use_linear_lr_decay and
            self.epoch > self.config.linear_lr_decay_epoch_start
        ):
            util.update_learning_rate(
                self.D_A, self.learning_rate_decrements['D']
            )
            util.update_learning_rate(
                self.D_B, self.learning_rate_decrements['D']
            )
            util.update_learning_rate(
                self.G, self.learning_rate_decrements['G']
            )

    def mean_epoch_losses(self):
        if self.losses['D']:
            D_loss_mean = np.mean(
                self.losses['D'][-self.data['num_train_batches']:]
            )
            G_loss_mean = np.mean([
                x['loss']
                for x in self.losses['G'][-self.data['num_train_batches']:]
            ])
            reconstruction_loss_mean = np.mean(
                self.losses['reconstruction'][-self.data['num_train_batches']:]
            )
            D_A_loss_mean = np.mean(
                self.losses['D_A'][-self.data['num_train_batches']:]
            )
            D_B_loss_mean = np.mean(
                self.losses['D_B'][-self.data['num_train_batches']:]
            )
            return {
                'D': D_loss_mean,
                'G': G_loss_mean,
                'r': reconstruction_loss_mean,
                'DA': D_A_loss_mean,
                'DB': D_B_loss_mean
            }

    def handle_epoch_callbacks(self):
        epoch_progressbar_postfix = self.mean_epoch_losses()

        if self.epoch % self.config.save_interval_examples == 0:
            self.save_example_images()

        if self.epoch % self.config.save_interval_model == 0:
            for model in [self.D_A, self.D_B, self.G_A2B, self.G_B2A]:
                models.io.save_model(
                    model, self.epoch, self.result_paths.saved_models
                )
            # Cheap way to output custom information without interfering
            # with the progress bars.
            epoch_progressbar_postfix['z'] = "Models saved."

        models.io.save_losses(self.losses, self.result_paths.base)

        self.epoch_progress_bar.set_postfix(**epoch_progressbar_postfix)

    def save_example_images(self):
        for train_or_test in ['train', 'test']:
            real_A_images = self.data[f'{train_or_test}_A_image_examples']
            real_B_images = self.data[f'{train_or_test}_B_image_examples']
            synthetic_B_images = self.G_A2B.predict(real_A_images)
            synthetic_A_images = self.G_B2A.predict(real_B_images)
            reconstructed_A_images = self.G_B2A.predict(synthetic_B_images)
            reconstructed_B_images = self.G_A2B.predict(synthetic_A_images)
            for i in range(self.config.num_examples_to_track):
                self.save_image_triplet(
                    real_A_images[i],
                    synthetic_B_images[i],
                    reconstructed_A_images[i],
                    self.result_paths.__dict__[
                        f'examples_history_{train_or_test}_A'
                    ] / f'epoch{self.epoch:04d}_example{i}.png'
                )
                self.save_image_triplet(
                    real_B_images[i],
                    synthetic_A_images[i],
                    reconstructed_B_images[i],
                    self.result_paths.__dict__[
                        f'examples_history_{train_or_test}_B'
                    ] / f'epoch{self.epoch:04d}_example{i}.png'
                )

    def save_temporary_example_images(
        self, real_A_image, real_B_image, synthetic_A_image, synthetic_B_image
    ):
        reconstructed_A_images = self.G_B2A.predict(synthetic_B_image)
        reconstructed_B_images = self.G_A2B.predict(synthetic_A_image)

        real_images = np.vstack((real_A_image[0], real_B_image[0]))
        synthetic_images = np.vstack((synthetic_B_image[0], synthetic_A_image[0]))
        reconstructed_images = np.vstack((reconstructed_A_images[0], reconstructed_B_images[0]))

        self.save_image_triplet(
            real_images,
            synthetic_images,
            reconstructed_images,
            self.result_paths.examples_history / 'tmp.png'
        )

    def calculate_learning_rate_decrements(self):
        updates_per_epoch_D = (
            2 * self.data['num_train_batches'] +
            self.config.discriminator_iterations - 1
        )
        updates_per_epoch_G = (
            self.data['num_train_batches'] +
            self.config.generator_iterations - 1
        )
        if self.config.use_identity_learning:
            updates_per_epoch_G *= \
                (1 + 1 / self.config.identity_learning_modulus)
        denominator_D = (
            self.config.epochs - self.config.linear_lr_decay_epoch_start
        ) * updates_per_epoch_D
        denominator_G = (
            self.config.epochs - self.config.linear_lr_decay_epoch_start
        ) * updates_per_epoch_G
        self.learning_rate_decrements = {
            'D': self.config.learning_rate_D / denominator_D,
            'G': self.config.learning_rate_G / denominator_G
        }

    def save_image_triplet(self, real, synthetic, reconstructed, path_name):
        image = np.hstack((real, synthetic, reconstructed))
        if self.config.image_shape[-1] == 1:
            image = image[:, :, 0]
        models.io.pil_image_from_normalized_array(image).save(path_name)

    def load_generator_weights(self, model_key):
        self.result_paths = config.construct_result_paths(
            model_key=model_key
        )
        models.io.load_weights_for_model(
            self.G_A2B, self.result_paths.saved_models
        )
        models.io.load_weights_for_model(
            self.G_B2A, self.result_paths.saved_models
        )

    def generate_synthetic_images(self):

        def generate_synthetic_images_batch(batch):
            synthetic_B_images = self.G_A2B.predict(batch['A_images'])
            synthetic_A_images = self.G_B2A.predict(batch['B_images'])
            for i in range(len(synthetic_A_images)):
                save_image(
                    synthetic_A_images[i],
                    batch['B_image_paths'][i].stem + '_synthetic.png',
                    'A'
                )
            for i in range(len(synthetic_B_images)):
                save_image(
                    synthetic_B_images[i],
                    batch['A_image_paths'][i].stem + '_synthetic.png',
                    'B'
                )

        def save_image(image, name, domain):
            if self.config.image_shape[-1] == 1:
                image = image[:, :, 0]
            models.io.pil_image_from_normalized_array(image).save(
                self.result_paths.__dict__[
                    f'generated_synthetic_{domain}_images'
                ] / name
            )

        logging.info("Generating synthetic images for the entire test set.")

        if self.config.use_data_generator:
            for batch in tqdm.tqdm(
                self.data['test_batch_generator'],
                unit='batch'
            ):
                generate_synthetic_images_batch(batch)
        else:
            batch = {
                'A_image_paths': self.data['test_A_image_paths'],
                'B_image_paths': self.data['test_B_image_paths'],
                'A_images': self.data['test_A_images'],
                'B_images': self.data['test_B_images'],
            }
            generate_synthetic_images_batch(batch)
        logging.info(
            f"{len(self.data['test_B_image_paths'])} "
            "synthetic A images have been generated in "
            f"{self.result_paths.generated_synthetic_A_images}."
        )
        logging.info(
            f"{len(self.data['test_A_image_paths'])} "
            "synthetic B images have been generated in "
            f"{self.result_paths.generated_synthetic_B_images}."
        )
