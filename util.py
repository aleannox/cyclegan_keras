"Utility code which is potentially useful for other projects."


import json
import os
import pathlib

import IPython.display
import numpy as np
import tensorflow as tf


class CustomJSONEncoder(json.JSONEncoder):
    "Custom JSON encoder which can serialize additional types."
    def default(self, o):  # pylint: disable=method-hidden
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, pathlib.Path):
            return str(o)
        else:
            return super().default(o)


def set_tensorflow_verbosity(verbose):
    if verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        tf.compat.v1.logging.set_verbosity(
            tf.compat.v1.logging.DEBUG
        )
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.compat.v1.logging.set_verbosity(
            tf.compat.v1.logging.ERROR
        )


def set_tensorflow_speedup_options(use_xla, use_auto_mixed_precision):
    # https://www.tensorflow.org/xla
    tf.config.optimizer.set_jit(use_xla)
    # https://www.tensorflow.org/guide/graph_optimization
    # https://www.tensorflow.org/guide/keras/mixed_precision
    # only useful for compute capability >=7.0 (e.g. RTX, V100)
    tf.config.optimizer.set_experimental_options({
        'auto_mixed_precision': use_auto_mixed_precision
    })


class ReflectionPadding2D(tf.keras.layers.Layer):
    """Reflection padding taken from
    https://github.com/fastai/courses/blob/master/deeplearning2/neural-style.ipynb
    """
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [tf.keras.layers.InputSpec(ndim=4)]
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            input_shape[1] + 2 * self.padding[0],
            input_shape[2] + 2 * self.padding[1],
            input_shape[3]
        )

    def call(self, inputs, **kwargs):
        w_pad, h_pad = self.padding
        return tf.pad(
            inputs,
            [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]],
            'REFLECT'
        )

    def get_config(self):  # pylint: disable=useless-super-delegation
        # TODO: Add padding arguments to returned config.
        return super().get_config()


def identity_layer(name):
    "Identity layer - used for renaming."
    return tf.keras.layers.Lambda(lambda x: x, name=name)


def update_learning_rate(model, decrement):
    new_lr = max(
        tf.keras.backend.get_value(model.optimizer.lr) - decrement,
        0
    )
    tf.keras.backend.set_value(model.optimizer.lr, new_lr)


def plot_model_svg(model, **kwargs):
    """Plot a keras model as SVG in a Jupyter notebook.
    Useful if you want to rescale the plot with `dpi` while preserving
    legibility.

    (Almost) drop in replacement for `tensorflow.keras.utils.plot_model`.
    TODO: Add `to_file` parameter for saving.
    """
    return IPython.display.SVG(
        tf.keras.utils.model_to_dot(
            model,
            **kwargs
        ).create(prog='dot', format='svg')
    )


def pandas_series_to_numpy(series):
    """Fix series with multidimensional arrays as values which are
    not proper numpy arrays because Pandas mixes list and numpy.array."""
    return np.array(list(map(list, series.values)))
