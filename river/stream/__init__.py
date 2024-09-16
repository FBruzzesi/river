"""Streaming utilities.

The module includes tools to iterate over data streams.

"""

from __future__ import annotations

from .cache import Cache
from .iter_arff import iter_arff
from .iter_array import iter_array
from .iter_csv import iter_csv
from .iter_frame import iter_frame
from .iter_libsvm import iter_libsvm
from .qa import simulate_qa
from .shuffling import shuffle
from .tweet_stream import TwitterLiveStream
from .twitch_chat_stream import TwitchChatStream

__all__ = [
    "Cache",
    "iter_arff",
    "iter_array",
    "iter_csv",
    "iter_frame",
    "iter_libsvm",
    "simulate_qa",
    "shuffle",
    "TwitterLiveStream",
    "TwitchChatStream",
]

try:
    import polars  # noqa: F401

    from .iter_polars import iter_polars

    __all__ += ["iter_polars"]
except ImportError:
    pass

try:
    import pandas  # noqa: F401

    from .iter_pandas import iter_pandas

    __all__ += ["iter_pandas"]
except ImportError:
    pass

try:
    from .iter_sklearn import iter_sklearn_dataset

    __all__ += ["iter_sklearn_dataset"]
except ImportError:
    pass

try:
    from .iter_sql import iter_sql

    __all__ += ["iter_sql"]
except ImportError:
    pass

try:
    from .iter_vaex import iter_vaex

    __all__ += ["iter_vaex"]
except ImportError:
    pass
