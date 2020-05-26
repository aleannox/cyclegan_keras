import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

import config


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(D_Ata, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scipy.signal.lfilter(b, a, D_Ata)
    return y


def plot_losses(
    model_key,
    epochs=None,
    point_gap=1,
    figsize=(20, 7)
):
    D_A_losses = []
    D_B_losses = []
    G_A_D_losses_synthetic = []
    G_B_D_losses_synthetic = []
    G_A_losses_reconstructed = []
    G_B_losses_reconstructed = []
    D_losses = []
    G_losses = []
    reconstruction_losses = []

    with (
        config.STATIC_PATHS.results / model_key / 'meta_data.json'
    ).open() as json_file:
        meta_data = json.load(json_file)

    with (
        config.STATIC_PATHS.results / model_key / 'loss_output.csv'
    ).open(newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            D_A_losses.append(float(row['D_A_losses']))
            D_B_losses.append(float(row['D_B_losses']))
            G_A_D_losses_synthetic.append(float(row['G_A_D_losses_synthetic']))
            G_B_D_losses_synthetic.append(float(row['G_B_D_losses_synthetic']))
            G_A_losses_reconstructed.append(float(row['G_A_losses_reconstructed']))
            G_B_losses_reconstructed.append(float(row['G_B_losses_reconstructed']))
            D_losses.append(float(row['D_losses']))
            reconstruction_losses.append(float(row['reconstruction_losses']))
            G_loss = row['G_losses']
            if G_loss[0] == '[':
                G_loss = G_loss.split(',')[0][1:]
            G_losses.append(float(G_loss))

        if epochs:
            num_points_max = epochs * meta_data['num_train_images_max']
        else:
            num_points_max = len(D_A_losses)

        # Calculate interesting things to plot
        D_A_losses = np.array(D_A_losses)[:num_points_max]
        D_B_losses = np.array(D_B_losses)[:num_points_max]
        G_A_losses = (
            np.array(G_A_D_losses_synthetic) +
            np.array(G_A_losses_reconstructed)
        )[:num_points_max]
        G_B_losses = (
            np.array(G_B_D_losses_synthetic) +
            np.array(G_B_losses_reconstructed)
        )[:num_points_max]
        R_A_losses = np.array(G_A_losses_reconstructed)[:num_points_max]
        R_B_losses = np.array(G_B_losses_reconstructed)[:num_points_max]

        G_losses = np.array(G_losses)[:num_points_max]
        D_losses = np.array(D_losses)[:num_points_max]
        reconstruction_losses = (
            np.array(G_A_losses_reconstructed) +
            np.array(G_B_losses_reconstructed)
        )[:num_points_max]

    points = range(0, len(G_losses), point_gap)
    fs = 1000
    cutoff = 2
    order = 6

    # Lowpass filter
    G_A = butter_lowpass_filter(G_A_losses[points], cutoff, fs, order)
    G_B = butter_lowpass_filter(G_B_losses[points], cutoff, fs, order)

    D_A = butter_lowpass_filter(D_A_losses[points], cutoff, fs, order)
    D_B = butter_lowpass_filter(D_B_losses[points], cutoff, fs, order)

    R_A = butter_lowpass_filter(R_A_losses[points], cutoff, fs, order)
    R_B = butter_lowpass_filter(R_B_losses[points], cutoff, fs, order)

    G = butter_lowpass_filter(G_losses[points], cutoff, fs, order)
    D = butter_lowpass_filter(D_losses[points], cutoff, fs, order)
    R = butter_lowpass_filter(reconstruction_losses[points], cutoff, fs, order)

    x = np.array(points) / meta_data['num_train_images_max']

    _, axs = plt.subplots(2, 2, figsize=figsize)

    ax = axs[0][0]
    ax.plot(x, G_A, label='G_A')
    ax.plot(x, G_B, label='G_B')
    ax.set_xlabel('epochs')
    ax.set_ylabel('generator losses')
    ax.legend()

    ax = axs[0][1]
    ax.plot(x, D_A, label='D_A')
    ax.plot(x, D_B, label='D_B')
    ax.set_xlabel('epochs')
    ax.set_ylabel('discriminator losses')
    ax.legend()

    ax = axs[1][0]
    ax.plot(x, R_A, label='A')
    ax.plot(x, R_B, label='B')
    ax.set_xlabel('epochs')
    ax.set_ylabel('reconstruction losses')
    ax.legend()

    ax = axs[1][1]
    ax.plot(x, G, label='G')
    ax.plot(x, D, label='D')
    ax.plot(x, R, label='reconstruction')
    ax.set_xlabel('epochs')
    ax.set_ylabel('losses')
    ax.legend()

    plt.tight_layout()
