"Model agnostic code."


def add_common_parser_arguments(parser):
    "Add common arguments for TensorFlow control."

    parser.add_argument(
        '--use-xla', dest='use_xla', action='store_true',
        help="Use XLA (Accelerated Linear Algebra) TensorFlow compiler."
    )
    parser.add_argument(
        '--dont-use-xla', dest='use_xla', action='store_false',
        help="Don't use XLA (Accelerated Linear Algebra) TensorFlow compiler."
    )
    parser.set_defaults(use_xla=False)
    parser.add_argument(
        '--use-auto-mixed-precision',
        dest='use_auto_mixed_precision', action='store_true',
        help="Use auto mixed precision Keras API speedup."
    )
    parser.add_argument(
        '--dont-use-auto-mixed-precision',
        dest='use_auto_mixed_precision', action='store_false',
        help="Don't use auto mixed precision Keras API speedup."
    )
    parser.set_defaults(use_auto_mixed_precision=False)
    parser.add_argument(
        '--verbose-tensorflow', dest='verbose_tensorflow', action='store_true',
        help="Do not suppress TensorFlow logging output."
    )
    parser.add_argument(
        '--silent-tensorflow', dest='verbose_tensorflow', action='store_false',
        help="Suppress TensorFlow logging output."
    )
    parser.set_defaults(verbose_tensorflow=False)

    return parser
