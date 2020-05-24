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


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return y


def plot_losses(model_key, point_gap=1):
    DA_losses = []
    DB_losses = []
    gA_d_losses_synthetic = []
    gB_d_losses_synthetic = []
    gA_losses_reconstructed = []
    gB_losses_reconstructed = []
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
            DA_losses.append(float(row['DA_losses']))
            DB_losses.append(float(row['DB_losses']))
            gA_d_losses_synthetic.append(float(row['gA_d_losses_synthetic']))
            gB_d_losses_synthetic.append(float(row['gB_d_losses_synthetic']))
            gA_losses_reconstructed.append(float(row['gA_losses_reconstructed']))
            gB_losses_reconstructed.append(float(row['gB_losses_reconstructed']))
            D_losses.append(float(row['D_losses']))
            reconstruction_losses.append(float(row['reconstruction_losses']))
            G_loss = row['G_losses']
            if G_loss[0] == '[':
                G_loss = G_loss.split(',')[0][1:]
            G_losses.append(float(G_loss))

        # Calculate interesting things to plot
        DA_losses = np.array(DA_losses)
        DB_losses = np.array(DB_losses)
        GA_losses = np.add(np.array(gA_d_losses_synthetic), np.array(gA_losses_reconstructed))
        GB_losses = np.add(np.array(gB_d_losses_synthetic), np.array(gB_losses_reconstructed))
        RA_losses = np.array(gA_losses_reconstructed)
        RB_losses = np.array(gB_losses_reconstructed)

        G_losses = np.array(G_losses)
        D_losses = np.array(D_losses)
        reconstruction_losses = np.add(
            np.array(gA_losses_reconstructed),
            np.array(gB_losses_reconstructed)
        )

    points = range(0, len(G_losses), point_gap)
    fs = 1000
    cutoff = 2
    order = 6

    # Lowpass filter
    GA = butter_lowpass_filter(GA_losses[points], cutoff, fs, order)
    GB = butter_lowpass_filter(GB_losses[points], cutoff, fs, order)

    DA = butter_lowpass_filter(DA_losses[points], cutoff, fs, order)
    DB = butter_lowpass_filter(DB_losses[points], cutoff, fs, order)

    RA = butter_lowpass_filter(RA_losses[points], cutoff, fs, order)
    RB = butter_lowpass_filter(RB_losses[points], cutoff, fs, order)

    G = butter_lowpass_filter(G_losses[points], cutoff, fs, order)
    D = butter_lowpass_filter(D_losses[points], cutoff, fs, order)
    R = butter_lowpass_filter(reconstruction_losses[points], cutoff, fs, order)

    x = np.array(points) / meta_data['num_train_examples']

    plt.figure(1)
    plt.plot(x, GA, label='GA_losses')
    plt.plot(x, GB, label='GB_losses')
    plt.xlabel('epochs')
    plt.ylabel('generator losses')
    plt.legend()

    plt.figure(2)
    plt.plot(x, DA, label='DA_losses')
    plt.plot(x, DB, label='DB_losses')
    plt.xlabel('epochs')
    plt.ylabel('discriminator losses')
    plt.legend()

    plt.figure(3)
    plt.plot(x, RA, label='reconstruction_loss_A')
    plt.plot(x, RB, label='reconstruction_loss_B')
    plt.xlabel('epochs')
    plt.ylabel('reconstruction losses')
    plt.legend()

    plt.figure(4)
    plt.plot(x, G, label='G_losses')
    plt.plot(x, D, label='D_losses')
    plt.plot(x, R, label='reconstruction losses')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.legend()
