import argparse
import logging

import config
import models
import models.autoencoder
import util


def get_arguments():
    parser = argparse.ArgumentParser(
        usage=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config-path', required=True,
        help=f"JSON config path, relative to {config.STATIC_PATHS.configs}."
    )
    parser.add_argument(
        '--model-key', required=True,
        help=(
            f"Load model from this key relative to {config.STATIC_PATHS.results}. "
            "The chronologically last model checkpoint is loaded."
        )
    )
    parser = models.add_common_parser_arguments(parser)
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arguments = get_arguments()
    logging.info("Running with the following arguments.")
    logging.info(arguments)
    model_config = config.autoencoder_config_from_json(arguments.config_path)
    logging.info("Running with the following config.")
    logging.info(model_config)
    util.set_tensorflow_verbosity(arguments.verbose_tensorflow)
    util.set_tensorflow_speedup_options(
        use_xla=arguments.use_xla,
        use_auto_mixed_precision=arguments.use_auto_mixed_precision
    )

    try:
        model = models.autoencoder.AutoEncoder(model_config)
        model.load_weights(arguments.model_key)
        model.prepare_data()
        model.encode()
    except KeyboardInterrupt:
        logging.info("Aborting encoding.")
