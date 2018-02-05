"""Robot JSON Encoder."""
from json.encoder import JSONEncoder
from typing import Any

import numpy as np  # type: ignore


class RobotJSONEncoder(JSONEncoder):
    """Robot JSON Encoder class."""

    def default(self, o: Any) -> Any:
        """
        Return serializable robot objects.

        :param o:
        :return:
        """
        # process np arrays
        if isinstance(o, np.ndarray):
            return o.tolist()

        # process np scalar types
        try:
            if str(o.dtype) in np.typeDict:  # pragma: no branch
                return str(o)
        except AttributeError:
            pass

        # break down object into a dict if possible
        try:
            o = o.__dict__
        except AttributeError:
            pass
        else:
            return o

        # let the base class default method raise the TypeError
        # https://docs.python.org/3/library/json.html
        return JSONEncoder.default(self, o)
