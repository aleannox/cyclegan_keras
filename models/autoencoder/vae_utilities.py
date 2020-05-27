"Utilities for VAE."


import logging
import time

import numpy as np
import matplotlib.pyplot as plt
import MulticoreTSNE
import sklearn.decomposition
import tensorflow as tf


# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = tf.keras.backend.shape(z_mean)[0]
    dim = tf.keras.backend.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon


def plot_results(model_pair, args, imgs):

    encoder, decoder = model_pair
    x = imgs

    # display a 2D plot of the layout classes in the latent space
    z_mean, _, _ = encoder.predict(x, batch_size=args.batch_size)

    # select 2 axes of highest variation
    # better / todo: PCA and then inverse rotation
    stds = np.std(z_mean, axis=0)
    best_dims = np.argpartition(-stds, 1)[:2]  # argpartition sorts min first

    basename = f'vae_{args.model_type}_{args.encoding_dim}'

    def scatterplot(z, suffix=''):
        plt.figure(figsize=(20, 20))
        plt.scatter(z[:, 0], z[:, 1], alpha=0.5, s=0.1, c=[.5, .5, .5])
        plt.xlabel(f'z{suffix}[0]')
        plt.ylabel(f'z{suffix}[1]')
        plt.title(f'{args.model_type}_{args.encoding_dim}{suffix}')
        plt.savefig(
            args.result_paths.base / f'{basename}_scatter{suffix}.png'
        )

    # PCA scatterplot
    pca = sklearn.decomposition.PCA(n_components=2)
    z_pca = pca.fit(z_mean).transform(z_mean)
    scatterplot(z_pca, '_pca')

    if args.tsne:
        # tsne scatterplot
        logging.info("computing TSNE ...")
        start = time.time()
        tsne = MulticoreTSNE.MulticoreTSNE(
            n_components=2, random_state=42, n_jobs=8
        )
        z_tsne = tsne.fit_transform(z_mean)
        logging.info(f"... finished in {time.time() - start:.0f}s")
        scatterplot(z_tsne, '_tsne')

    # display a n x n 2D manifold of layouts
    n = 5
    figure = np.zeros((args.height * n, args.width * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of layout classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.zeros((1,) + z_mean.shape[1:])
            z_sample[0, best_dims] = [xi, yi]
            x_decoded = decoder.predict(z_sample)
            layout = x_decoded[0].reshape(args.height, args.width)
            figure[
                i * args.height: (i + 1) * args.height,
                j * args.width: (j + 1) * args.width
            ] = layout

    plt.figure(figsize=(10, 10))
    start_range_h = args.height // 2
    end_range_h = n * args.height + start_range_h + 1
    pixel_range_h = np.arange(start_range_h, end_range_h, args.height)
    start_range_w = args.width // 2
    end_range_w = n * args.width + start_range_w + 1
    pixel_range_w = np.arange(start_range_w, end_range_w, args.width)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range_w, sample_range_x)
    plt.yticks(pixel_range_h, sample_range_y)
    plt.xlabel(f'z[{best_dims[0]}]')
    plt.ylabel(f'z[{best_dims[1]}]')
    plt.title(args.model_type)
    plt.imshow(figure, cmap=plt.cm.binary)
    plt.savefig(
        args.result_paths.base / f'{basename}_gen.png'
    )
