"Model-agnostic IO."


import json
import logging
import pickle
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

    random.seed(4242)

    data = {}
    for tt in ['train', 'test']:
        data[f'{tt}_image_paths'] = {}
        data[f'{tt}_images'] = {}
        data[f'{tt}_image_examples'] = {}
        for domain in config.DOMAINS:
            data[f'{tt}_image_paths'][domain] = sorted(
                (
                    config.STATIC_PATHS.source_images /
                    model_config.dataset_name / f'{tt}_{domain}'
                ).glob(ALLOWED_IMAGE_PATTERN)
            )
            if model_config.__dict__[f'num_{tt}_{domain}_images']:
                data[f'{tt}_image_paths'][domain] = \
                    data[f'{tt}_image_paths'][domain][
                        :model_config.__dict__[f'num_{tt}_{domain}_images']
                    ]
            if data[f'{tt}_image_paths'][domain]:
                data[f'{tt}_image_examples'][domain] = create_image_array(
                    random.sample(
                        data[f'{tt}_image_paths'][domain],
                        model_config.num_examples_to_track
                    ),
                    model_config.image_shape
                )
            if not model_config.use_data_generator:
                data[f'{tt}_images'][domain] = create_image_array(
                    data[f'{tt}_image_paths'][domain],
                    model_config.image_shape
                )
        if model_config.use_data_generator:
            data[f'{tt}_batch_generator'] = BothDomainsDataGenerator(
                data[f'{tt}_image_paths'],
                batch_size=model_config.batch_size,
                image_shape=model_config.image_shape,
                shuffle=tt == 'train'
            )
        data[f'num_{tt}_images_max'] = max(
            len(data[f'{tt}_image_paths'][domain])
            for domain in config.DOMAINS
        )
        data[f'num_{tt}_batches'] = compute_num_batches(
            data[f'num_{tt}_images_max'],
            model_config.batch_size
        )
    return data


def load_single_domain_data(model_config):
    data = {}
    # Load train and test data from train image folder and make the split
    # here on the fly.
    if model_config.domain == 'both':
        data['all_image_paths'] = []
        for domain in config.DOMAINS:
            data['all_image_paths'] += sorted(
                (
                    config.STATIC_PATHS.source_images /
                    model_config.dataset_name / f'train_{domain}'
                ).glob(ALLOWED_IMAGE_PATTERN)
            )
    else:
        data['all_image_paths'] = sorted(
            (
                config.STATIC_PATHS.source_images /
                model_config.dataset_name / f'train_{model_config.domain}'
            ).glob(ALLOWED_IMAGE_PATTERN)
        )
    random.seed(4242)
    random.shuffle(data['all_image_paths'])
    data['train_image_paths'] = \
        data['all_image_paths'][model_config.num_test_images:]
    data['test_image_paths'] = \
        data['all_image_paths'][:model_config.num_test_images]
    for tt in ['train', 'test']:
        data[f'{tt}_image_examples'] = create_image_array(
            random.sample(
                data[f'{tt}_image_paths'],
                model_config.num_examples_to_track
            ),
            model_config.image_shape
        )
        if model_config.use_data_generator and tt == 'train':
            # We do not use a generator for the test set.
            data[f'{tt}_batch_generator'] = SingleDomainDataGenerator(
                data[f'{tt}_image_paths'],
                batch_size=model_config.batch_size,
                image_shape=model_config.image_shape,
                shuffle=True
            )
        else:
            data[f'{tt}_images'] = create_image_array(
                data[f'{tt}_image_paths'],
                model_config.image_shape
            )
    return data


class BothDomainsDataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        image_paths,
        image_shape,
        batch_size,
        shuffle
    ):
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.image_paths = image_paths
        self.shuffle = shuffle

    def __len__(self):
        return compute_num_batches(
            max(
                len(self.image_paths[domain])
                for domain in config.DOMAINS
            ),
            self.batch_size
        )

    def __getitem__(self, idx):
        batch = {
            'image_paths': {},
            'images': {}
        }
        for domain, other in config.DOMAIN_PAIRS:
            batch['image_paths'][domain] = get_batch(
                idx,
                self.batch_size,
                self.image_paths[domain],
                len(self.image_paths[other])
            )
            batch['images'][domain] = create_image_array(
                batch['image_paths'][domain], self.image_shape
            )
        return batch

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.image_paths)


class SingleDomainDataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        image_paths,
        image_shape,
        batch_size,
        shuffle
    ):
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.image_paths = image_paths
        self.shuffle = shuffle

    def __len__(self):
        return compute_num_batches(
            len(self.image_paths),
            self.batch_size
        )

    def __getitem__(self, idx):
        image_paths_batch = get_batch(
            idx,
            self.batch_size,
            self.image_paths,
            len(self.image_paths)
        )
        return create_image_array(
            image_paths_batch, self.image_shape
        )

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.image_paths)


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
            if image.shape == image_shape[:-1]:
                # Black and white image, but we want RGB.
                image = black_and_white_to_rgb(image)
            else:
                raise ValueError(
                    f"Not able to correct wrong image shape {image.shape} "
                    f"to target shape {image_shape}."
                )
        image_array.append(normalize_array(image))
    return np.array(image_array).astype('float32')


def black_and_white_to_rgb(bw_image_array):
    return np.stack(
        tuple(bw_image_array for _ in range(3)),
        axis=-1
    )


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


def save_losses(losses, result_paths_base):
    with (result_paths_base / 'losses_history.pickle').open('wb') as file:
        pickle.dump(losses, file)


def save_image_tuple(images, path_name, num_channels):
    image = np.hstack(images)
    if num_channels == 1:
        image = image[:, :, 0]
    pil_image_from_normalized_array(image).save(
        path_name, quality=75
    )


def load_weights_for_model(model, result_paths_saved_models):
    if not result_paths_saved_models.exists():
        raise FileNotFoundError(f"{result_paths_saved_models} does not exist.")
    path_to_weights = sorted(
        result_paths_saved_models.glob(f'{model.name}_weights_epoch_*.hdf5')
    )[-1]  # Load last availdomainle weights.
    logging.info("Loading weights from %s.", path_to_weights)
    model.load_weights(str(path_to_weights))


def compute_num_batches(num_examples, batch_size):
    return int(
        math.ceil(
            num_examples / batch_size
        )
    )
