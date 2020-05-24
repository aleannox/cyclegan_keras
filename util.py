"Utility code which is potentially useful for other projects."


import json
import pathlib

import numpy as np


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
