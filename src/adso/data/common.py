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
    extension: str = "txt",
) -> Dataset:
    path = Path(path)
    if label:
        if lines:
            return LabelledDataset(
                list(
                    chain.from_iterable(
                        map(
                            lambda f: map(
                                lambda line: (line, f.stem),
                                f.read_text().splitlines(),
                            ),
                            path.glob("**/*." + extension),
                        )
                    )
                )
            )
        else:  # label and not lines
            return LabelledDataset(
                list(
                    map(
                        lambda f: (f.read_text(), f.parent.name),
                        path.glob("**/*." + extension),
                    )
                )
            )
    else:  # not label
        if lines:
            return Dataset(
                list(
                    chain.from_iterable(
                        map(
                            lambda f: f.read_text().splitlines(),
                            path.glob("**/*." + extension),
                        )
                    )
                )
            )
        else:  # not label and not lines
            return Dataset(
                list(map(lambda f: f.read_text(), path.glob("**/*." + extension)))
            )
