import logging
import random

import numpy as np
import tensorflow as tf

import config
import models.autoencoder.components as components
import models.io


class AutoEncoder():
    def __init__(self, model_config):
        self.config = model_config

        random.seed(4242)
        np.random.seed(4242)
        tf.random.set_seed(4242)

        self.prepare_model()

        # Attributes which are optionally filled later.
        self.result_paths = None
        self.data = None

    def prepare_model(self):
        if self.config.model_type == 'mlp':
            self.encoder = components.mlp_encoder(
                self.config.image_shape,
                self.config.encoding_dimension,
                self.config.intermediate_dimension
            )
            self.decoder = components.mlp_decoder(
                self.config.image_shape,
                self.config.encoding_dimension,
                self.config.intermediate_dimension
            )
        else:  # CNN
            pass

        inputs = tf.keras.Input(
            shape=self.config.image_shape,
            name='input'
        )
        z = self.encoder(inputs)
        outputs = self.decoder(z)

        self.model = tf.keras.Model(inputs, outputs, name='autoencoder')

        reconstruction_loss = tf.keras.losses.mse(inputs, outputs)
        reconstruction_loss *= np.prod(self.config.image_shape)
        # Loss to keep the latent encoding small.
        # Corresponds to KL divergence for variational autoencoder
        # without sampling (i.e. with zero variance).
        z_loss = 0.5 * tf.keras.backend.sum(tf.keras.backend.square(z), axis=-1)
        total_loss = (
            tf.keras.backend.mean(reconstruction_loss) +
            tf.keras.backend.mean(z_loss)
        )
        # Adding the loss in this way rather than via compile allows us to
        # avoid passing the inputs as targets explicitely.
        self.model.add_loss(total_loss)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.learning_rate,
                beta_1=self.config.adam_beta_1,
                beta_2=self.config.adam_beta_2
            )
        )

    def prepare_data(self):
        # Input data.
        if self.config.use_data_generator:
            logging.info("Using data generator.")
        else:
            logging.info("Caching data in RAM.")
        self.data = models.io.load_single_domain_data(self.config)
        logging.info("Data prepared.")

    def train(self):
        self.result_paths = config.construct_result_paths(
            create_dirs=True
        )
        logging.info(f"Saving metadata in {self.result_paths.base}.")
        models.io.save_metadata(
            data=self.config.__dict__,
            result_paths_base=self.result_paths.base
        )

        fit_args = dict(
            epochs=self.config.epochs,
            validation_data=(self.data['test_images'], None),
            callbacks=self.prepare_callbacks()
        )
        if self.config.use_data_generator:
            fit_args['x'] = self.data['train_batch_generator']
            fit_args['steps_per_epoch'] = len(
                self.data['train_batch_generator']
            )
        else:
            fit_args['x'] = self.data['train_images']
            fit_args['batch_size'] = self.config.batch_size

        self.model.fit(**fit_args)
        self.model.save_weights(
            str(self.result_paths.saved_models / 'final_weights.hdf5')
        )

    def prepare_callbacks(self):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.result_paths.tensorboard,
            histogram_freq=0,
            write_graph=False,
            write_grads=False,
            write_images=False,
            profile_batch=0
        )

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            str(self.result_paths.saved_models / 'best_weights.hdf5'),
            monitor='val_loss', verbose=1,
            save_best_only=True, mode='min'
        )

        examples_callback = components.SaveExamplesCallback(
            save_interval=self.config.save_interval_examples,
            real_examples={
                train_or_test: self.data[f'{train_or_test}_image_examples']
                for train_or_test in ['train', 'test']
            },
            sink_folders={
                train_or_test: self.result_paths.__dict__[
                    f'examples_history_{train_or_test}'
                ][self.config.domain]
                for train_or_test in ['train', 'test']
            },
            num_channels=self.config.image_shape[-1]
        )

        return [
            tensorboard_callback,
            checkpoint_callback,
            examples_callback
        ]
