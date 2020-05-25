import argparse
import logging

import config
import models
import models.cyclegan
import util


def get_arguments():
    parser = argparse.ArgumentParser(
        usage=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config-path',
        help=f"JSON config path, relative to {config.STATIC_PATHS.configs}."
    )
    parser = models.add_common_parser_arguments(parser)
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arguments = get_arguments()
    logging.info("Running with the following arguments.")
    logging.info(arguments)
    config_ = config.model_config_from_json(arguments.config_path)
    logging.info("Running with the following config.")
    logging.info(config_)
    util.set_tensorflow_verbosity(arguments.verbose_tensorflow)
    util.set_tensorflow_speedup_options(
        use_xla=arguments.use_xla,
        use_auto_mixed_precision=arguments.use_auto_mixed_precision
    )

    models.cyclegan.CycleGAN(
        config_=config_,
        generate_synthetic_images=False
    )
