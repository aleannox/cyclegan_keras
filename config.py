import dataclasses
import datetime
import json
import pathlib

import dacite


CONFIG_CONVERTERS = {
    pathlib.Path: pathlib.Path,
    tuple: tuple,
    bool: bool,
    int: int
}


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

STATIC_PATHS = StaticPaths(base=pathlib.Path('data'))


@dataclasses.dataclass
class ResultPaths:
    base: pathlib.Path
    saved_models: pathlib.Path = dataclasses.field(init=False)
    output_history_samples: pathlib.Path = dataclasses.field(init=False)
    output_history_samples_train_A: pathlib.Path = dataclasses.field(init=False)
    output_history_samples_train_B: pathlib.Path = dataclasses.field(init=False)
    output_history_samples_test_A: pathlib.Path = dataclasses.field(init=False)
    output_history_samples_test_B: pathlib.Path = dataclasses.field(init=False)
    generated_synthetic_images_A: pathlib.Path = dataclasses.field(init=False)
    generated_synthetic_images_B: pathlib.Path = dataclasses.field(init=False)
    def __post_init__(self):
        self.saved_models = self.base / 'saved_models'
        self.output_history_samples = self.base / 'output_history_samples'
        self.output_history_samples_train_A = self.output_history_samples / 'train_A'
        self.output_history_samples_train_B = self.output_history_samples / 'train_B'
        self.output_history_samples_test_A = self.output_history_samples / 'test_A'
        self.output_history_samples_test_B = self.output_history_samples / 'test_B'
        self.generated_synthetic_images_A = self.base / 'generated_synthetic_images' / 'A'
        self.generated_synthetic_images_B = self.base / 'generated_synthetic_images' / 'B'


def construct_result_paths(model_key=None):
    if model_key is None:
        model_key = datetime.datetime.now().isoformat(timespec='seconds')
    result_paths = dacite.from_dict(
        data_class=ResultPaths,
        data={'base': STATIC_PATHS.results / model_key},
        config=dacite.Config(type_hooks=CONFIG_CONVERTERS)
    )
    for path in result_paths.__dict__.values():
        path.mkdir(exist_ok=True, parents=True)
    return result_paths


@dataclasses.dataclass
class TrainConfig:
    source_images: pathlib.Path
    image_shape: tuple
    use_data_generator: bool
    batch_size: int


def train_config_from_json(config_path):
    with pathlib.Path(STATIC_PATHS.configs / config_path).open() as file:
        json_config = json.load(file)
    return dacite.from_dict(
        data_class=TrainConfig,
        data=json_config,
        config=dacite.Config(type_hooks=CONFIG_CONVERTERS)
    )
