"Model-agnostic IO."


import csv
import json
import logging

import numpy as np
import PIL.Image
import tensorflow as tf

import config
import util


def load_data(
    source_images,
    num_channels=3,
    batch_size=1,
    num_train_A_images=None,
    num_train_B_images=None,
    num_test_A_images=None,
    num_test_B_images=None,
    return_generator=True
):
    train_A_path = config.STATIC_PATHS.source_images / source_images / 'train_A'
    train_B_path = config.STATIC_PATHS.source_images / source_images / 'train_B'
    test_A_path = config.STATIC_PATHS.source_images / source_images / 'test_A'
    test_B_path = config.STATIC_PATHS.source_images / source_images / 'test_B'

    train_A_image_names = sorted(train_A_path.iterdir())
    if num_train_A_images:
        train_A_image_names = train_A_image_names[:num_train_A_images]

    train_B_image_names = sorted(train_B_path.iterdir())
    if num_train_B_images:
        train_B_image_names = train_B_image_names[:num_train_B_images]

    test_A_image_names = sorted(test_A_path.iterdir())
    if num_test_A_images:
        test_A_image_names = test_A_image_names[:num_test_A_images]

    test_B_image_names = sorted(test_B_path.iterdir())
    if num_test_B_images:
        test_B_image_names = test_B_image_names[:num_test_B_images]

    if return_generator:
        train_AB_images_generator = data_sequence(
            train_A_image_names,
            train_B_image_names,
            batch_size=batch_size
        )
        test_AB_images_generator = data_sequence(
            test_A_image_names,
            test_B_image_names,
            batch_size=batch_size
        )
        train_A_images = None
        train_B_images = None
        test_A_images = None
        test_B_images = None
    else:
        train_AB_images_generator = None
        test_AB_images_generator = None
        train_A_images = create_image_array(train_A_image_names, num_channels)
        train_B_images = create_image_array(train_B_image_names, num_channels)
        test_A_images = create_image_array(test_A_image_names, num_channels)
        test_B_images = create_image_array(test_B_image_names, num_channels)

    return {
        "train_A_images": train_A_images,
        "train_B_images": train_B_images,
        "test_A_images": test_A_images,
        "test_B_images": test_B_images,
        "train_A_image_names": train_A_image_names,
        "train_B_image_names": train_B_image_names,
        "test_A_image_names": test_A_image_names,
        "test_B_image_names": test_B_image_names,
        "train_AB_images_generator": train_AB_images_generator,
        "test_AB_images_generator": test_AB_images_generator
    }


ALLOWED_SUFFIXES = ['.jpg', '.jpeg', '.png']


def create_image_array(image_list, num_channels):
    image_array = []
    for image_name in image_list:
        if image_name.suffix.lower() in ALLOWED_SUFFIXES:
            if num_channels == 1:  # Gray scale image
                image = np.array(PIL.Image.open(image_name))
                image = image[:, :, np.newaxis]
            else:
                image = np.array(PIL.Image.open(image_name))
            image = normalize_array(image)
            if image.shape[-1] == num_channels:
                image_array.append(image)

    return np.array(image_array)


# If using 16 bit depth images, use the formula 'array = array / 32767.5 - 1' instead
def normalize_array(array):
    array = array / 127.5 - 1
    return array


class data_sequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        image_list_A,
        image_list_B,
        batch_size=1
    ):
        self.batch_size = batch_size
        self.data_A = []
        self.data_B = []
        for image_name in image_list_A:
            if image_name.suffix.lower() in ALLOWED_SUFFIXES:
                self.data_A.append(image_name)
        for image_name in image_list_B:
            if image_name.suffix.lower() in ALLOWED_SUFFIXES:
                self.data_B.append(image_name)

    def __len__(self):
        max_num_images = max(len(self.data_A), len(self.data_B))
        return (
            max_num_images // self.batch_size + 1
            if max_num_images % self.batch_size > 0
            else max_num_images // self.batch_size
        )

    def __getitem__(self, idx):
        if idx >= min(len(self.data_A), len(self.data_B)):
            # If all images soon are used for one domain,
            # randomly pick from this domain
            if len(self.data_A) <= len(self.data_B):
                indexes_A = np.random.randint(len(self.data_A), size=self.batch_size)
                batch_A = []
                for i in indexes_A:
                    batch_A.append(self.data_A[i])
                batch_B = self.data_B[idx * self.batch_size:(idx + 1) * self.batch_size]
            else:
                indexes_B = np.random.randint(len(self.data_B), size=self.batch_size)
                batch_B = []
                for i in indexes_B:
                    batch_B.append(self.data_B[i])
                batch_A = self.data_A[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            batch_A = self.data_A[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_B = self.data_B[idx * self.batch_size:(idx + 1) * self.batch_size]

        real_images_A = create_image_array(batch_A, 3)
        real_images_B = create_image_array(batch_B, 3)

        return {
            'real_images_A': real_images_A,
            'real_images_B': real_images_B,
            'real_image_paths_A': batch_A,
            'real_image_paths_B': batch_B
        }


def save_model(model, epoch, result_paths_saved_models):
    model_path_m = result_paths_saved_models / f'{model.name}_model.json'
    # ^ Architecture constant accross epochs.
    model_path_w = result_paths_saved_models / f'{model.name}_weights_epoch_{epoch:04d}.hdf5'
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
