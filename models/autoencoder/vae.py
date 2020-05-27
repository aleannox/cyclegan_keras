"""Variational autoencoder.
MLP and CNN variants.

Refactored from:
https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder_deconv.py
"""


import argparse
import logging
import random
import shutil

import tensorflow as tf

import config
import models
import models.autoencoder.components as components
import models.autoencoder.vae_utilities as vae_utilities
import models.io
import util


def main(args):
    setupname = f'{args.model_type}_{args.encoding_dim}'
    args.result_paths = config.construct_result_paths(
        create_dirs=True
    )
    args.domain = 'A'
    model_config = config.model_config_from_json(
        'panzer_schaeferhund_autoencoder.json'
    )
    args.batch_size = model_config.batch_size
    args.epochs = model_config.epochs
    args.width = model_config.image_shape[1]
    args.height = model_config.image_shape[0]
    args.num_channels = model_config.image_shape[2]
    data = models.io.load_data(model_config)

    random.seed(4242)
    x_full_paths = data['train_image_paths'][args.domain]
    random.shuffle(x_full_paths)
    x_train_paths = x_full_paths[:-1000]
    x_val_paths = x_full_paths[-1000:]
    x_full = models.io.SingleDomainDataGenerator(
        x_full_paths,
        model_config.image_shape,
        args.batch_size
    )
    x_train = models.io.SingleDomainDataGenerator(
        x_train_paths,
        model_config.image_shape,
        args.batch_size
    )
    x_val = models.io.create_image_array(
        x_val_paths, model_config.image_shape
    )

    logging.info(f"Training on {len(x_train_paths)} samples.")
    logging.info(f"Validating on {len(x_val_paths)} samples.")
    logging.info(f"Val shape {x_val.shape}.")

    # if args.model_type == 'mlp':
    encoder = components.mlp_encoder(
        model_config.image_shape,
        args.encoding_dim
    )
    decoder = components.mlp_decoder(
        model_config.image_shape,
        args.encoding_dim
    )
    # instantiate VAE model
    inputs = tf.keras.Input(shape=model_config.image_shape, name='vae_input')
    # outputs = decoder(encoder(inputs)[2])
    z = encoder(inputs)
    outputs = decoder(z)
    vae = tf.keras.Model(inputs, outputs, name=f'vae_{args.model_type}')

    reconstruction_loss = tf.keras.losses.mse(inputs, outputs)
    reconstruction_loss *= args.height * args.width * args.num_channels
    # kl_loss = -0.5 * tf.keras.backend.sum(
    #     (
    #         1 + z_log_var -
    #         tf.keras.backend.square(z_mean) -
    #         tf.keras.backend.exp(z_log_var)
    #     ),
    #     axis=-1
    # )
    # vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)
    kl_loss = tf.keras.backend.sum(tf.keras.backend.square(z), axis=-1)
    vae_loss = tf.keras.backend.mean(reconstruction_loss) + tf.keras.backend.mean(kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=tf.keras.optimizers.Adam())
    # tf.keras.utils.plot_model(
    #     vae,
    #     to_file=args.result_paths.base / f'vae_{args.model_type}.png',
    #     expand_nested=True,
    #     show_shapes=True
    # )

    weights_path = args.result_paths.saved_models / f'vae_{setupname}.hdf5'
    weights_path_best = weights_path.with_suffix('.best.hdf5')
    if args.weights:
        if args.weights == 'default':
            vae.load_weights(str(weights_path))
        elif args.weights == 'best':
            vae.load_weights(str(weights_path_best))
        else:
            vae.load_weights(
                str(
                    args.result_paths.saved_models / args.weights
                )
            )
    else:
        # train the autoencoder
        log_dir = args.result_paths.base / 'tensorboard' / setupname
        shutil.rmtree(log_dir, ignore_errors=True)
        tbcb = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=0,
            write_graph=False,
            write_grads=False,
            write_images=False,
            profile_batch=0
        )
        cpcb = tf.keras.callbacks.ModelCheckpoint(
            str(weights_path_best),
            monitor='val_loss', verbose=1,
            save_best_only=True, mode='min'
        )
        vae.fit(
            x_train,
            epochs=args.epochs,
            steps_per_epoch=models.io.compute_num_batches(
                len(x_train_paths),
                args.batch_size
            ),
            validation_data=(x_val, None),
            callbacks=[tbcb, cpcb]
        )
        vae.save_weights(str(weights_path))

    vae_utilities.plot_results((encoder, decoder), args, x_full)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights',
        help="Load hdf5 model trained weights"
    )
    parser.add_argument(
        '--model-type', default='mlp',
        help="Select model type: {'mlp' (default), 'cnn'}"
    )
    parser.add_argument(
        '--encoding-dim', default=2, type=int,
        help="Encoding dimension"
    )
    parser.add_argument(
        '--tsne', default=False, type=bool,
        help="Draw TSNE scatterplot."
    )
    parser = models.add_common_parser_arguments(parser)
    arguments = parser.parse_args()

    logging.info("Running with the following arguments.")
    logging.info(arguments)

    util.set_tensorflow_verbosity(arguments.verbose_tensorflow)
    util.set_tensorflow_speedup_options(
        use_xla=arguments.use_xla,
        use_auto_mixed_precision=arguments.use_auto_mixed_precision
    )

    try:
        main(arguments)
    except KeyboardInterrupt:
        logging.info("Aborting training.")
