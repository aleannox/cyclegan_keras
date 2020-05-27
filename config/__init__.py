import dataclasses
import datetime
import json
import pathlib
import typing

import dacite
import tensorflow as tf
import tensorflow_addons


CONFIG_CONVERTERS = {
    pathlib.Path: pathlib.Path,
    tuple: tuple,
    bool: bool,
    int: int,
    float: float,
}


NORMALIZATIONS = {
    'instance_normalization': tensorflow_addons.layers.InstanceNormalization
}

DOMAINS = ('A', 'B')
DOMAIN_PAIRS = tuple(zip(DOMAINS, DOMAINS[::-1]))

@dataclasses.dataclass
class StaticPaths:
    base: pathlib.Path
    configs: pathlib.Path = dataclasses.field(init=False)
    source_images: pathlib.Path = dataclasses.field(init=False)
    results: pathlib.Path = dataclasses.field(init=False)
    def __post_init__(self):
        self.configs = self.base / 'configs'
        self.source_images = self.base / 'source_images'
        self.results = self.base / 'results'

STATIC_PATHS = StaticPaths(
    base=(pathlib.Path(__file__).parents[1] / 'data').resolve()
)


@dataclasses.dataclass
class ResultPaths:
    base: pathlib.Path
    saved_models: pathlib.Path = dataclasses.field(init=False)
    examples_history: pathlib.Path = dataclasses.field(init=False)
    examples_history_train: typing.Dict[str, pathlib.Path] \
        = dataclasses.field(init=False)
    examples_history_test: typing.Dict[str, pathlib.Path] \
        = dataclasses.field(init=False)
    generated_synthetic_images: typing.Dict[str, pathlib.Path] \
        = dataclasses.field(init=False)
    tensorboard: pathlib.Path = dataclasses.field(init=False)
    def __post_init__(self):
        self.saved_models = self.base / 'saved_models'
        self.examples_history = self.base / 'examples_history'
        self.examples_history_train = {
            domain: self.examples_history / f'train_{domain}'
            for domain in DOMAINS
        }
        self.examples_history_test = {
            domain: self.examples_history / f'test_{domain}'
            for domain in DOMAINS
        }
        self.generated_synthetic_images = {
            domain: self.base / 'generated_synthetic_images' / domain
            for domain in DOMAINS
        }
        self.tensorboard = self.base / 'tensorboard'


def construct_result_paths(model_key=None, create_dirs=False):
    if model_key is None:
        model_key = datetime.datetime.now().isoformat(timespec='seconds')
    result_paths = dacite.from_dict(
        data_class=ResultPaths,
        data={'base': STATIC_PATHS.results / model_key}
    )
    if create_dirs:
        for path_or_path_dict in result_paths.__dict__.values():
            if isinstance(path_or_path_dict, pathlib.Path):
                path_or_path_dict.mkdir(exist_ok=True, parents=True)
            else:
                for path in path_or_path_dict.values():
                    path.mkdir(exist_ok=True, parents=True)
    return result_paths


@dataclasses.dataclass
class CycleGANConfig:
    dataset_name: pathlib.Path
    image_shape: tuple
    use_data_generator: bool
    epochs: int
    batch_size: int
    save_interval_examples: int
    save_interval_temporary_examples: int
    save_interval_model: int
    normalization: str
    lambda_ABA: float
    lambda_BAB: float
    lambda_D: float
    learning_rate_D: float
    learning_rate_G: float
    generator_iterations: int
    discriminator_iterations: int
    adam_beta_1: float
    adam_beta_2: float
    num_examples_to_track: int
    synthetic_pool_size: int
    use_linear_lr_decay: bool
    linear_lr_decay_epoch_start: int
    use_identity_learning: bool
    identity_learning_modulus: int
    use_patchgan_discriminator: bool
    use_multiscale_discriminator: bool
    use_resize_convolution: bool
    use_supervised_learning: bool
    supervised_learning_weight: float
    real_discriminator_label: float
    num_train_A_images: typing.Union[int, None]
    num_train_B_images: typing.Union[int, None]
    num_test_A_images: typing.Union[int, None]
    num_test_B_images: typing.Union[int, None]


@dataclasses.dataclass
class AutoEncoderConfig:
    dataset_name: pathlib.Path
    image_shape: tuple
    use_data_generator: bool
    epochs: int
    batch_size: int
    save_interval_examples: int
    domain: str
    model_type: str
    encoding_dimension: int
    intermediate_dimension: int
    num_examples_to_track: int
    num_test_images: int
    learning_rate: float
    adam_beta_1: float
    adam_beta_2: float
    cnn_levels: int
    cnn_kernel_size: int
    cnn_filters: int
    cnn_strides: int
    cnn_padding: str


def cyclegan_config_from_json(config_path):
    return model_config_from_json(config_path, data_class=CycleGANConfig)


def autoencoder_config_from_json(config_path):
    return model_config_from_json(config_path, data_class=AutoEncoderConfig)


def model_config_from_json(config_path, data_class):
    with pathlib.Path(STATIC_PATHS.configs / config_path).open() as file:
        json_config = json.load(file)
    return dacite.from_dict(
        data_class=data_class,
        data=json_config,
        config=dacite.Config(type_hooks=CONFIG_CONVERTERS)
    )
