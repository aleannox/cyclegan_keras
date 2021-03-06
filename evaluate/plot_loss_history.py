import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

import config
import models.io


FILTER_FS = 1000
FILTER_CUTOFF = 2
FILTER_ORDER = 6
RAW_ALPHA = 0.2
DOMAIN_COLORS = dict(zip(config.DOMAINS, ('red', 'blue')))


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
    min_epoch=1,
    max_epoch=None,
    point_gap=1,
    figsize=(20, 7),
    log=False,
    **plot_kws
):
    with (
        config.STATIC_PATHS.results / model_key / 'meta_data.json'
    ).open() as json_file:
        meta_data = json.load(json_file)

    with (
        config.STATIC_PATHS.results / model_key / 'losses_history.pickle'
    ).open('rb') as file:
        losses = pickle.load(file)

    num_batches_per_epoch = models.io.compute_num_batches(
        meta_data['num_train_images_max'],
        meta_data['batch_size']
    )

    num_points_min = min_epoch * num_batches_per_epoch
    num_points_max = len(losses['G'])
    if max_epoch:
        num_points_max = min(
            num_points_max,
            max_epoch * num_batches_per_epoch
        )

    # Calculate interesting things to plot
    plot_data = {}
    for domain in config.DOMAINS:
        plot_data[f'D_{domain}'] = np.array(losses[f'D_{domain}'])
        plot_data[f'G_{domain}'] = np.array(losses[f'G_{domain}'])
        plot_data[f'R_{domain}'] = np.array(losses[f'G_{domain}_reconstructed'])
    plot_data['D'] = np.array(losses['D'])
    plot_data['G'] = np.array([x['loss'] for x in losses['G']])
    plot_data['R'] = sum(
        plot_data[f'R_{domain}'] for domain in config.DOMAINS
    )

    points = range(num_points_min, num_points_max, point_gap)

    # Lowpass filter
    for kind in ['D', 'G', 'R']:
        for variant in (None,) + config.DOMAINS:
            key = f'{kind}_{variant}' if variant else kind
            try:
                plot_data[f'{key}_raw'] = plot_data[key][points]
                plot_data[f'{key}_filtered'] = butter_lowpass_filter(
                    plot_data[key],
                    FILTER_CUTOFF,
                    FILTER_FS // 200 * num_batches_per_epoch,
                    FILTER_ORDER
                )[points]
            except IndexError:
                print(key)
                print(points)
                print(plot_data[key])
                raise

    x = np.array(points) / num_batches_per_epoch

    def plot_kind(kind, ax):
        for domain, color in DOMAIN_COLORS.items():
            ax.plot(
                x, plot_data[f'{kind}_{domain}_filtered'],
                label=f'{domain} filtered', c=color,
                **plot_kws
            )
            ax.plot(
                x, plot_data[f'{kind}_{domain}_raw'],
                label=f'{domain}', alpha=RAW_ALPHA, c=color,
                **plot_kws
            )
        ax.set_xlabel('epoch')
        ax.set_ylabel(f'{kind} losses')
        if log:
            ax.set_yscale('log')
        ax.legend()

    _, axs = plt.subplots(2, 2, figsize=figsize)

    ax = axs[0][0]
    plot_kind('G', ax)

    ax = axs[0][1]
    plot_kind('D', ax)

    ax = axs[1][0]
    plot_kind('R', ax)

    ax = axs[1][1]
    for kind, color in zip(
        ('G', 'D', 'R'),
        ('red', 'blue', 'green')
    ):
        ax.plot(
            x, plot_data[f'{kind}_filtered'],
            label=f'{kind} filtered', c=color,
            **plot_kws
        )
        ax.plot(
            x, plot_data[f'{kind}_raw'],
            label=kind, c=color, alpha=RAW_ALPHA,
            **plot_kws
        )
        if log:
            ax.set_yscale('log')
    ax.set_xlabel('epoch')
    ax.set_ylabel('losses')
    ax.legend()

    plt.tight_layout()
