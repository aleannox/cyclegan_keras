"Model-agnostic IO."


import csv
import json
import logging
import math
import random

import numpy as np
import PIL.Image
import tensorflow as tf

import config
import util


# Match JPG and PNG files with `glob`.
ALLOWED_IMAGE_PATTERN = '*.[jJpP][pPnN]*[gG]'


def load_data(model_config):
    # TODO: While we need to correct for a size imbalance of A and B datasets
    # for the training set, we don't need to do this for the test set.
    # We are currently doing this if using the data generator.
    # The only negative effect of doing so is running image generation multiple
    # times for some images in the generation mode, which is inefficient.
    data = {}
    for tt in ['train', 'test']:
        for ab in ['A', 'B']:
            data[f'{tt}_{ab}_image_paths'] = sorted(
                (
                    config.STATIC_PATHS.source_images /
                    model_config.dataset_name / f'{tt}_{ab}'
                ).glob(ALLOWED_IMAGE_PATTERN)
            )
            if model_config.__dict__[f'num_{tt}_{ab}_images']:
                data[f'{tt}_{ab}_image_paths'] = \
                    data[f'{tt}_{ab}_image_paths'][
                        :model_config.__dict__[f'num_{tt}_{ab}_images']
                    ]
            data[f'{tt}_{ab}_image_examples'] = create_image_array(
                random.sample(
                    data[f'{tt}_{ab}_image_paths'],
                    model_config.num_examples_to_track
                ),
                model_config.image_shape
            )
            if not model_config.use_data_generator:
                data[f'{tt}_{ab}_images'] = create_image_array(
                    data[f'{tt}_{ab}_image_paths'],
                    model_config.image_shape
                )
        if model_config.use_data_generator:
            data[f'{tt}_batch_generator'] = DataGenerator(
                data[f'{tt}_A_image_paths'],
                data[f'{tt}_B_image_paths'],
                batch_size=model_config.batch_size,
                image_shape=model_config.image_shape
            )
        data[f'num_{tt}_images_max'] = max(
            len(data[f'{tt}_A_image_paths']),
            len(data[f'{tt}_B_image_paths'])
        )
        data[f'num_{tt}_batches'] = compute_num_batches(
            data[f'num_{tt}_images_max'],
            model_config.batch_size
        )
    return data


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        A_image_paths,
        B_image_paths,
        image_shape,
        batch_size
    ):
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.A_image_paths = A_image_paths
        self.B_image_paths = B_image_paths

    def __len__(self):
        return compute_num_batches(
            max(len(self.A_image_paths), len(self.B_image_paths)),
            self.batch_size
        )

    def __getitem__(self, idx):
        A_image_paths_batch = get_batch(
            idx, self.batch_size, self.A_image_paths, len(self.B_image_paths)
        )
        B_image_paths_batch = get_batch(
            idx, self.batch_size, self.B_image_paths, len(self.A_image_paths)
        )
        return {
            'A_image_paths': A_image_paths_batch,
            'B_image_paths': B_image_paths_batch,
            'A_images': create_image_array(
                A_image_paths_batch, self.image_shape
            ),
            'B_images': create_image_array(
                B_image_paths_batch, self.image_shape
            ),
        }


def get_batch(
    batch_index,
    batch_size,
    images,
    num_other_images
):
    """Fill batch with random samples in order to match other dataset in case it
    has more samples and we are in a batch affected by this.
    """
    batch_slice = slice(
        batch_index * batch_size, (batch_index + 1) * batch_size
    )
    num_image_batches = compute_num_batches(len(images), batch_size)
    if (
        len(images) < num_other_images and
        batch_index >= num_image_batches - 1
    ):
        # TODO: Use a more memory efficient solution.
        if isinstance(images, np.ndarray):
            indices_enlarged = list(range(len(images))) + random.sample(
                range(len(images)), num_other_images - len(images)
            )
            images_enlarged = images[indices_enlarged]
        else:
            images_enlarged = images + random.sample(
                images, num_other_images - len(images)
            )
        return images_enlarged[batch_slice]
    else:
        return images[batch_slice]


def create_image_array(image_paths, image_shape):
    image_array = []
    for image_path in image_paths:
        if image_shape[-1] == 1:  # Gray scale image
            image = np.array(PIL.Image.open(image_path))
            image = image[:, :, np.newaxis]
        else:
            image = np.array(PIL.Image.open(image_path))
        if image.shape != image_shape:
            raise ValueError(
                f"{image_path} has the wrong shape: "
                f"{image.shape} instead of {image_shape}."
            )
        image_array.append(image)
    return normalize_array(np.array(image_array))


def normalize_array(array):
    "Normalize 8bit image array to max/min = ±1"
    # If using 16 bit depth images, use 'array = array / 32767.5 - 1' instead
    array = array / 127.5 - 1
    return array


def pil_image_from_normalized_array(image_array):
    "Create 8bit image from array with max/min = ±1"
    return PIL.Image.fromarray(
        ((image_array + 1) * 127.5).astype('uint8')
    )


def save_model(model, epoch, result_paths_saved_models):
    model_path_m = result_paths_saved_models / f'{model.name}_model.json'
    # ^ Architecture constant accross epochs.
    model_path_w = (
        result_paths_saved_models /
        f'{model.name}_weights_epoch_{epoch:04d}.hdf5'
    )
    model.save_weights(str(model_path_w))
    model_path_m.write_text(model.to_json())


def save_metadata(data, result_paths_base):
    with (result_paths_base / 'meta_data.json').open('w') as outfile:
        json.dump(
            data, outfile,
            sort_keys=True, indent=4, cls=util.CustomJSONEncoder
        )


def save_losses(history, result_paths_base):
    keys = sorted(history.keys())
    with (result_paths_base / 'loss_output.csv').open('w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(keys)
        writer.writerows(zip(*[history[key] for key in keys]))


def load_weights_for_model(model, result_paths_saved_models):
    if not result_paths_saved_models.exists():
        raise FileNotFoundError(f"{result_paths_saved_models} does not exist.")
    path_to_weights = sorted(
        result_paths_saved_models.glob(f'{model.name}_weights_epoch_*.hdf5')
    )[-1]  # Load last available weights.
    logging.info("Loading weights from %s.", path_to_weights)
    model.load_weights(str(path_to_weights))


def compute_num_batches(num_examples, batch_size):
    return int(
        math.ceil(
            num_examples / batch_size
        )
    )
