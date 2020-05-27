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
        self.D = {}
        loss_weights_D = [0.5]
        # ^ Correct for D having twice as many interations as G due to
        # training on real and synthetic images.
        for domain in config.DOMAINS:
            if self.config.use_multiscale_discriminator:
                self.D[domain] = components.multi_scale_discriminator(
                    **discriminator_args, name=f'D_{domain}'
                )
                loss_weights_D *= 2
            else:
                self.D[domain] = components.discriminator(
                    **discriminator_args, name=f'D_{domain}'
                )
            self.D[domain].compile(
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
        self.G_single = {}
        for domain, other in config.DOMAIN_PAIRS:
            self.G_single[domain] = components.generator(
                **generator_args, name=f'G_{other}2{domain}'
            )
            if self.config.use_identity_learning:
                self.G_single[domain].compile(
                    optimizer=self.optimizer_G, loss=tf.keras.losses.mae
                )

    def prepare_full_model(self):
        inputs = []
        real = {
            domain: tf.keras.Input(
                shape=self.config.image_shape, name=f'real_{domain}'
            )
            for domain in config.DOMAINS
        }
        synthetic = {}
        for domain, other in config.DOMAIN_PAIRS:
            synthetic[domain] = util.identity_layer(
                f'synthetic_{domain}'
            )(self.G_single[domain](real[other]))
        D_guess_synthetic = {}
        reconstructed = {}
        outputs = []
        losses = {}
        loss_weights = {}

        for domain, other in config.DOMAIN_PAIRS:
            # Inputs
            inputs.append(real[domain])

            # Output candidates, wrapped in identities for renaming.
            if self.config.use_multiscale_discriminator:
                D_guess_synthetic[domain] = []
                for i in range(2):
                    D_guess_synthetic[domain].append(
                        util.identity_layer(f'D_{domain}_guess_synthetic_{i}')(
                            self.D[domain](synthetic[domain])[i]
                        )
                    )
            else:
                D_guess_synthetic[domain] = util.identity_layer(
                    f'D_{domain}_guess_synthetic'
                )(self.D[domain](synthetic[domain]))
            reconstructed[domain] = \
                util.identity_layer(f'reconstructed_{domain}')(
                    self.G_single[domain](synthetic[other])
                )

            # Outputs, losses, weights.

            # Cyclic generators
            outputs.append(reconstructed[domain])
            losses[f'reconstructed_{domain}'] = tf.keras.losses.mae
            loss_weights[f'reconstructed_{domain}'] = self.config.__dict__[
                f'lambda_{domain}{other}{domain}'
            ]

            # Discriminators A, B
            if self.config.use_multiscale_discriminator:
                for i in range(2):
                    outputs.append(D_guess_synthetic[domain][i])
                    losses[f'D_{domain}_guess_synthetic_{i}'] = \
                        tf.keras.losses.mse
                    loss_weights[f'D_{domain}_guess_synthetic_{i}'] = \
                        self.config.lambda_D
            else:
                outputs.append(D_guess_synthetic[domain])
                losses[f'D_{domain}_guess_synthetic'] = tf.keras.losses.mse
                loss_weights[f'D_{domain}_guess_synthetic'] = \
                    self.config.lambda_D

            # Supervised individual generators B -> A, A -> B
            if self.config.use_supervised_learning:
                outputs.append(synthetic[domain])
                losses[f'synthetic_{domain}'] = tf.keras.losses.mae
                loss_weights[f'synthetic_{domain}'] = \
                    self.config.supervised_learning_weight

        self.G = tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name='G'
        )
        # Compile G with frozen discriminators, we do not want to change their
        # weights in the generator training.
        for domain in config.DOMAINS:
            self.D[domain].trainable = False
        self.G.compile(
            optimizer=self.optimizer_G,
            loss=losses,
            loss_weights=loss_weights
        )
        # Not necessary if the discriminators have been already compiled.
        # (Which is the case at the time of writing.)
        for domain in config.DOMAINS:
            self.D[domain].trainable = True

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
                label_shape = (
                    (self.config.batch_size,) +
                    self.D[config.DOMAINS[0]].output_shape[i][1:]
                )
                self.data['positive_discriminator_labels'].append(
                    np.ones(shape=label_shape) *
                    self.config.real_discriminator_label
                )
                self.data['negative_discriminator_labels'].append(
                    np.zeros(shape=label_shape)
                )
        else:
            label_shape = (
                (self.config.batch_size,) +
                self.D[config.DOMAINS[0]].output_shape[1:]
            )
            self.data['positive_discriminator_labels'] = (
                np.ones(shape=label_shape) *
                self.config.real_discriminator_label
            )
            self.data['negative_discriminator_labels'] = \
                np.zeros(shape=label_shape)

        # Image pools used to update the discriminators.
        self.data['synthetic_pool'] = {
            domain: components.ImagePool(self.config.synthetic_pool_size)
            for domain in config.DOMAINS
        }

        logging.info("Data prepared.")

    def train(self):
        self.result_paths = config.construct_result_paths(
            create_dirs=True
        )
        logging.info(f"Saving metadata in {self.result_paths.base}.")
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

        self.initialize_losses()

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

    def initialize_losses(self):
        self.losses = {}
        for domain in config.DOMAINS:
            self.losses[f'D_{domain}'] = []
            self.losses[f'G_{domain}_D_synthetic'] = []
            self.losses[f'G_{domain}_reconstructed'] = []
            self.losses[f'G_{domain}'] = []
            if self.config.use_identity_learning:
                self.losses[f'G_{domain}_identity'] = []
        self.losses['reconstruction'] = []
        self.losses['D'] = []
        self.losses['G'] = []

    def run_data_generator_training_epoch(self):
        batch_progress_bar = tqdm.tqdm(
            self.data['train_batch_generator'],
            unit='batch'
        )
        for batch_index, batch in enumerate(batch_progress_bar):
            self.run_batch_training_iteration(batch_index, batch)

    def run_ram_training_epoch(self):
        random_orders = {
            domain: np.random.randint(
                len(self.data['train_images'][domain]),
                size=len(self.data['train_images'][domain])
            )
            for domain in config.DOMAINS
        }
        # If we want supervised learning, A and B must match.
        if self.config.use_supervised_learning:
            random_orders[config.DOMAINS[1]] = random_orders[config.DOMAINS[0]]
        batch_progress_bar = tqdm.trange(
            self.data['num_train_batches'], unit='batch'
        )
        for batch_index in batch_progress_bar:
            batch = {'images': {}}
            for domain, other in config.DOMAIN_PAIRS:
                batch['images'][domain] = models.io.get_batch(
                    batch_index,
                    self.config.batch_size,
                    self.data['train_images'][domain][random_orders[domain]],
                    len(self.data['train_images'][other])
                )
            self.run_batch_training_iteration(batch_index, batch)

    def run_batch_training_iteration(self, batch_index, batch):
        real_images = batch['images']
        D_loss = self.run_batch_training_iteration_discriminator(
            batch_index, real_images
        )
        G_loss = self.run_batch_training_iteration_generator(real_images)
        G_identity_loss = self.run_batch_training_iteration_identity(
            batch_index, real_images
        )
        self.record_losses(D_loss, G_loss, G_identity_loss)
        self.maybe_update_learning_rates()

    def run_batch_training_iteration_discriminator(
        self, batch_index, real_images
    ):
        synthetic_images = {}
        for domain, other in config.DOMAIN_PAIRS:
            synthetic_images[domain] = \
                self.G_single[domain].predict(real_images[other])
            synthetic_images[domain] = \
                self.data['synthetic_pool'][domain].query(
                    synthetic_images[domain]
                )

        D_loss = {}
        for _ in range(self.config.discriminator_iterations):
            for domain in config.DOMAINS:
                D_loss_real = self.D[domain].train_on_batch(
                    x=real_images[domain],
                    y=self.data['positive_discriminator_labels']
                )
                D_loss_synthetic = self.D[domain].train_on_batch(
                    x=synthetic_images[domain],
                    y=self.data['negative_discriminator_labels']
                )
                if self.config.use_multiscale_discriminator:
                    D_loss[domain] = sum(D_loss_real) + sum(D_loss_synthetic)
                else:
                    D_loss[domain] = D_loss_real + D_loss_synthetic

        if batch_index % self.config.save_interval_temporary_examples == 0:
            self.save_temporary_example_images(
                real_images,
                synthetic_images
            )

        return D_loss

    def run_batch_training_iteration_generator(self, real_images):
        x = {}
        y = {}

        for domain in config.DOMAINS:
            x[f'real_{domain}'] = real_images[domain]
            y[f'reconstructed_{domain}'] = real_images[domain]
            if self.config.use_multiscale_discriminator:
                for i in range(2):
                    y[f'D_{domain}_guess_synthetic_{i}'] = \
                        self.data['positive_discriminator_labels'][i]
            else:
                y[f'D_{domain}_guess_synthetic'] = \
                    self.data['positive_discriminator_labels']
            if self.config.use_supervised_learning:
                y[f'synthetic_{domain}'] = real_images[domain]

        for _ in range(self.config.generator_iterations):
            G_loss = self.G.train_on_batch(x=x, y=y, return_dict=True)

        return G_loss

    def run_batch_training_iteration_identity(
        self, batch_index, real_images
    ):
        if (
            self.config.use_identity_learning and
            batch_index % self.config.identity_learning_modulus == 0
        ):
            G_identity_loss = {}
            for domain in config.DOMAINS:
                G_identity_loss[domain] = self.G_single[domain].train_on_batch(
                    x=real_images[domain], y=real_images[domain]
                )
            return G_identity_loss

    def record_losses(self, D_loss, G_loss, G_identity_loss):
        for domain in config.DOMAINS:
            self.losses[f'D_{domain}'].append(D_loss[domain])
            if self.config.use_multiscale_discriminator:
                G_D_loss_synthetic = sum(
                    G_loss[f'D_{domain}_guess_synthetic_loss_{i}']
                    for i in range(2)
                )
            else:
                G_D_loss_synthetic = G_loss[f'D_{domain}_guess_synthetic_loss']
            self.losses[f'G_{domain}_D_synthetic'].append(G_D_loss_synthetic)
            self.losses[f'G_{domain}_reconstructed'].append(
                G_loss[f'reconstructed_{domain}_loss']
            )
            self.losses[f'G_{domain}'].append(
                G_D_loss_synthetic + G_loss[f'reconstructed_{domain}_loss']
            )
            if self.config.use_identity_learning:
                self.losses[f'G_{domain}_identity'].append(
                    G_identity_loss[domain]
                )
        self.losses['D'].append(sum(D_loss.values()))
        self.losses['G'].append(G_loss)
        self.losses['reconstruction'].append(
            sum(
                G_loss[f'reconstructed_{domain}_loss']
                for domain in config.DOMAINS
            )
        )

    def maybe_update_learning_rates(self):
        if (
            self.config.use_linear_lr_decay and
            self.epoch > self.config.linear_lr_decay_epoch_start
        ):
            for domain in config.DOMAINS:
                util.update_learning_rate(
                    self.D[domain], self.learning_rate_decrements['D']
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
            for models_ in [self.D.values(), self.G_single.values()]:
                for model in models_:
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
            for domain, other in config.DOMAIN_PAIRS:
                real_images = \
                    self.data[f'{train_or_test}_image_examples'][domain]
                synthetic_images = self.G_single[other].predict(real_images)
                reconstructed_images = self.G_single[domain].predict(synthetic_images)
                for i in range(self.config.num_examples_to_track):
                    models.io.save_image_tuple(
                        (
                            real_images[i],
                            synthetic_images[i],
                            reconstructed_images[i]
                        ),
                        self.result_paths.__dict__[
                            f'examples_history_{train_or_test}'
                        ][domain] / f'epoch{self.epoch:04d}_example{i}.jpg',
                        self.config.image_shape[-1]
                    )

    def save_temporary_example_images(self, real_images, synthetic_images):
        reconstructed_images = {}
        for domain, other in config.DOMAIN_PAIRS:
            reconstructed_images[domain] = \
                self.G_single[domain].predict(synthetic_images[other])

        real_images_stacked = np.vstack(
            tuple(images[0] for images in real_images.values())
        )
        synthetic_images_stacked = np.vstack(
            tuple(images[0] for images in synthetic_images.values())[::-1]
        )
        reconstructed_images_stacked = np.vstack(
            tuple(images[0] for images in reconstructed_images.values())
        )

        self.save_image_triplet(
            real_images_stacked,
            synthetic_images_stacked,
            reconstructed_images_stacked,
            self.result_paths.examples_history / 'tmp.jpg'
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

    def load_generator_weights(self, model_key):
        self.result_paths = config.construct_result_paths(
            model_key=model_key
        )
        for domain in config.DOMAINS:
            models.io.load_weights_for_model(
                self.G_single[domain], self.result_paths.saved_models
            )

    def generate_synthetic_images(self):

        def generate_synthetic_images_batch(batch):
            for domain, other in config.DOMAIN_PAIRS:
                synthetic_images = \
                    self.G_single[domain].predict(batch['images'][other])
                for i in range(len(synthetic_images)):
                    save_image(
                        synthetic_images[i],
                        batch['image_paths'][other][i].stem + '_synthetic.jpg',
                        domain
                    )

        def save_image(image, name, domain):
            if self.config.image_shape[-1] == 1:
                image = image[:, :, 0]
            models.io.pil_image_from_normalized_array(image).save(
                self.result_paths.generated_synthetic_images[domain] / name,
                quality=75
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
                key: self.data[f'test_{key}']
                for key in ('images', 'image_paths')
            }
            generate_synthetic_images_batch(batch)

        for domain, other in config.DOMAIN_PAIRS:
            logging.info(
                f"{len(self.data['test_image_paths'][other])} "
                f"synthetic {domain} images have been generated in "
                f"{self.result_paths.generated_synthetic_images[domain]}."
            )
