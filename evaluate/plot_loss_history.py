import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

import config


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return y


def plot_losses(
    model_key,
    epochs=None,
    point_gap=1,
    figsize=(20, 7)
):
    with (
        config.STATIC_PATHS.results / model_key / 'meta_data.json'
    ).open() as json_file:
        meta_data = json.load(json_file)

    with (
        config.STATIC_PATHS.results / model_key / 'losses_history.pickle'
    ).open('rb') as file:
        losses = pickle.load(file)

    if epochs:
        num_points_max = epochs * meta_data['num_train_images_max']
    else:
        num_points_max = len(losses['D_A'])

    # Calculate interesting things to plot
    D_A_losses = np.array(losses['D_A'])[:num_points_max]
    D_B_losses = np.array(losses['D_B'])[:num_points_max]
    G_A_losses = (
        np.array(losses['G_A_D_synthetic']) +
        np.array(losses['G_A_reconstructed'])
    )[:num_points_max]
    G_B_losses = (
        np.array(losses['G_B_D_synthetic']) +
        np.array(losses['G_B_reconstructed'])
    )[:num_points_max]
    R_A_losses = np.array(losses['G_A_reconstructed'])[:num_points_max]
    R_B_losses = np.array(losses['G_B_reconstructed'])[:num_points_max]

    G_losses = np.array([x['loss'] for x in losses['G']])[:num_points_max]
    D_losses = np.array(losses['D'])[:num_points_max]
    reconstruction_losses = R_A_losses + R_B_losses

    points = range(0, len(G_losses), point_gap)
    fs = 1000
    cutoff = 2
    order = 6
    raw_alpha = 0.2

    # Lowpass filter
    G_A_raw = G_A_losses[points]
    G_A = butter_lowpass_filter(G_A_raw, cutoff, fs, order)
    G_B_raw = G_B_losses[points]
    G_B = butter_lowpass_filter(G_B_raw, cutoff, fs, order)

    D_A_raw = D_A_losses[points]
    D_A = butter_lowpass_filter(D_A_raw, cutoff, fs, order)
    D_B_raw = D_B_losses[points]
    D_B = butter_lowpass_filter(D_B_raw, cutoff, fs, order)

    R_A_raw = R_A_losses[points]
    R_A = butter_lowpass_filter(R_A_raw, cutoff, fs, order)
    R_B_raw = R_A_losses[points]
    R_B = butter_lowpass_filter(R_B_raw, cutoff, fs, order)

    G_raw = G_losses[points]
    G = butter_lowpass_filter(G_raw, cutoff, fs, order)
    D_raw = D_losses[points]
    D = butter_lowpass_filter(D_raw, cutoff, fs, order)
    R_raw = reconstruction_losses[points]
    R = butter_lowpass_filter(R_raw, cutoff, fs, order)

    x = np.array(points) / meta_data['num_train_images_max']

    _, axs = plt.subplots(2, 2, figsize=figsize)

    ax = axs[0][0]
    ax.plot(x, G_A, label='G_A filtered', c='red')
    ax.plot(x, G_B, label='G_B filtered', c='blue')
    ax.plot(x, G_A_raw, label='G_A', alpha=raw_alpha, c='red')
    ax.plot(x, G_B_raw, label='G_B', alpha=raw_alpha, c='blue')
    ax.set_xlabel('epochs')
    ax.set_ylabel('generator losses')
    ax.legend()

    ax = axs[0][1]
    ax.plot(x, D_A, label='D_A filtered', c='red')
    ax.plot(x, D_B, label='D_B filtered', c='blue')
    ax.plot(x, D_A_raw, label='D_A', alpha=raw_alpha, c='red')
    ax.plot(x, D_B_raw, label='D_B', alpha=raw_alpha, c='blue')
    ax.set_xlabel('epochs')
    ax.set_ylabel('discriminator losses')
    ax.legend()

    ax = axs[1][0]
    ax.plot(x, R_A, label='A filtered', c='red')
    ax.plot(x, R_B, label='B filtered', c='blue')
    ax.plot(x, R_A_raw, label='A', alpha=raw_alpha, c='red')
    ax.plot(x, R_B_raw, label='B', alpha=raw_alpha, c='blue')
    ax.set_xlabel('epochs')
    ax.set_ylabel('reconstruction losses')
    ax.legend()

    ax = axs[1][1]
    ax.plot(x, G, label='G filtered', c='red')
    ax.plot(x, D, label='D filtered', c='blue')
    ax.plot(x, R, label='reconstruction filtered', c='green')
    ax.plot(x, G_raw, label='G', alpha=raw_alpha, c='red')
    ax.plot(x, D_raw, label='D', alpha=raw_alpha, c='blue')
    ax.plot(x, R_raw, label='reconstruction', alpha=raw_alpha, c='green')
    ax.set_xlabel('epochs')
    ax.set_ylabel('losses')
    ax.legend()

    plt.tight_layout()
