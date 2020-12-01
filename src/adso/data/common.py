import os

from pathlib import Path
from itertools import chain
from typing import Union

from .dataset import Dataset, LabelledDataset
from ..common import ADSODIR

DATADIR = ADSODIR / "data"


def load_txt(
    path: Union[str, bytes, os.PathLike],
    lines: bool = False,
    label: bool = False,
    extension: Union[str, None] = "txt",
    encoding: str = "utf-8",
    ignore_error: bool = True,
) -> Dataset:
    path = Path(path)
    if extension is None:
        extension = ""
    else:
        extension = "." + extension

    if ignore_error:
        errors = "ignore"
    else:
        errors = None

    if label:
        if lines:
            return LabelledDataset(
                list(
                    chain.from_iterable(
                        map(
                            lambda f: map(
                                lambda line: (line, f.stem),
                                f.read_text(
                                    encoding=encoding, errors=errors
                                ).splitlines(),
                            ),
                            filter(
                                lambda f: f.is_file(), path.glob("**/*" + extension)
                            ),
                        )
                    )
                )
            )
        else:  # label and not lines
            return LabelledDataset(
                list(
                    map(
                        lambda f: (
                            f.read_text(encoding=encoding, errors=errors),
                            f.parent.name,
                        ),
                        filter(lambda f: f.is_file(), path.glob("**/*" + extension)),
                    )
                )
            )
    else:  # not label
        if lines:
            return Dataset(
                list(
                    chain.from_iterable(
                        map(
                            lambda f: f.read_text(
                                encoding=encoding, errors=errors
                            ).splitlines(),
                            filter(
                                lambda f: f.is_file(), path.glob("**/*" + extension)
                            ),
                        )
                    )
                )
            )
        else:  # not label and not lines
            return Dataset(
                list(
                    map(
                        lambda f: f.read_text(encoding=encoding, errors=errors),
                        filter(lambda f: f.is_file(), path.glob("**/*" + extension)),
                    )
                )
            )
