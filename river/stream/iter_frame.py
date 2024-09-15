from __future__ import annotations

import narwhals.stable.v1 as nw

from river import base, stream


def iter_frame(
    X: nw.IntoFrameT, y: nw.Series | nw.IntoFrameT | None = None, **kwargs
) -> base.typing.Stream:
    """Iterates over the rows of a `pandas.DataFrame`.

    Parameters
    ----------
    X
        A dataframe of features.
    y
        A series or a dataframe with one column per target.
    kwargs
        Extra keyword arguments are passed to the underlying call to `stream.iter_array`.

    Examples
    --------

    >>> import pandas as pd
    >>> from river import stream

    >>> X = pd.DataFrame({
    ...     'x1': [1, 2, 3, 4],
    ...     'x2': ['blue', 'yellow', 'yellow', 'blue'],
    ...     'y': [True, False, False, True]
    ... })
    >>> y = X.pop('y')

    >>> for xi, yi in stream.iter_pandas(X, y):
    ...     print(xi, yi)
    {'x1': 1, 'x2': 'blue'} True
    {'x1': 2, 'x2': 'yellow'} False
    {'x1': 3, 'x2': 'yellow'} False
    {'x1': 4, 'x2': 'blue'} True
    """

    X = nw.from_native(X, eager_only=True, strict=True)
    y = nw.from_native(y, eager_only=True, strict=False, allow_series=True)

    kwargs["feature_names"] = X.columns
    if isinstance(y, nw.DataFrame):
        kwargs["target_names"] = y.columns

    yield from stream.iter_array(X=X.to_numpy(), y=y if y is None else y.to_numpy(), **kwargs)
