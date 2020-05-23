import os

import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence

import config


def load_data(
    source_images,
    num_channels=3,
    batch_size=1,
    nr_A_train_imgs=None,
    nr_B_train_imgs=None,
    nr_A_test_imgs=None,
    nr_B_test_imgs=None,
    generator=True
):
    train_A_path = config.STATIC_PATHS.source_images / source_images / 'train_A'
    train_B_path = config.STATIC_PATHS.source_images / source_images / 'train_B'
    test_A_path = config.STATIC_PATHS.source_images / source_images / 'test_A'
    test_B_path = config.STATIC_PATHS.source_images / source_images / 'test_B'

    train_A_image_names = list(train_A_path.iterdir())
    if nr_A_train_imgs is not None:
        train_A_image_names = train_A_image_names[:nr_A_train_imgs]

    train_B_image_names = list(train_B_path.iterdir())
    if nr_B_train_imgs is not None:
        train_B_image_names = train_B_image_names[:nr_B_train_imgs]

    test_A_image_names = list(test_A_path.iterdir())
    if nr_A_test_imgs is not None:
        test_A_image_names = test_A_image_names[:nr_A_test_imgs]

    test_B_image_names = list(test_B_path.iterdir())
    if nr_B_test_imgs is not None:
        test_B_image_names = test_B_image_names[:nr_B_test_imgs]

    if generator:
        return data_sequence(train_A_path, train_B_path, train_A_image_names, train_B_image_names, batch_size=batch_size)  # D_model, use_multiscale_discriminator, use_supervised_learning, REAL_LABEL)
    else:
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
            "test_B_image_names": test_B_image_names
        }


ALLOWED_SUFFIXES = ['.jpg', '.jpeg', '.png']


def create_image_array(image_list, num_channels):
    image_array = []
    for image_name in image_list:
        if image_name.suffix.lower() in ALLOWED_SUFFIXES:
            if num_channels == 1:  # Gray scale image
                image = np.array(Image.open(image_name))
                image = image[:, :, np.newaxis]
            else:
                image = np.array(Image.open(image_name))
            image = normalize_array(image)
            if image.shape[-1] == num_channels:
                image_array.append(image)

    return np.array(image_array)

# If using 16 bit depth images, use the formula 'array = array / 32767.5 - 1' instead
def normalize_array(array):
    array = array / 127.5 - 1
    return array


class data_sequence(Sequence):

    def __init__(self, train_A_path, train_B_path, image_list_A, image_list_B, batch_size=1):
        self.batch_size = batch_size
        self.train_A = []
        self.train_B = []
        for image_name in image_list_A:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_A.append(os.path.join(train_A_path, image_name))
        for image_name in image_list_B:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_B.append(os.path.join(train_B_path, image_name))

    def __len__(self):
        return int(max(len(self.train_A), len(self.train_B)) / float(self.batch_size))

    def __getitem__(self, idx):
        print(idx)
        assert not isinstance(idx, str)
        if idx >= min(len(self.train_A), len(self.train_B)):
            # If all images soon are used for one domain,
            # randomly pick from this domain
            if len(self.train_A) <= len(self.train_B):
                indexes_A = np.random.randint(len(self.train_A), size=self.batch_size)
                batch_A = []
                for i in indexes_A:
                    batch_A.append(self.train_A[i])
                batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]
            else:
                indexes_B = np.random.randint(len(self.train_B), size=self.batch_size)
                batch_B = []
                for i in indexes_B:
                    batch_B.append(self.train_B[i])
                batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]

        real_images_A = create_image_array(batch_A, 3)
        real_images_B = create_image_array(batch_B, 3)

        return real_images_A, real_images_B
