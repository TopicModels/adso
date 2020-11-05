"""Datasets and datased related functions."""

from __future__ import annotations

from itertools import chain
from pathlib import Path
from warnings import warn

import tensorflow as tf

from adso import ADSODIR

# Create data folder if it not exists
DATADIR = ADSODIR / "data"
DATADIR.mkdir(exist_ok=True)


class Dataset:
    def __init__(self: Dataset, data: tf.Tensor):
        if tf.rank(data) != 1:
            raise ValueError("data must be a 1D tensor")
        self.data: tf.Tensor = data

    def get_data(self: Dataset) -> tf.Tensor:
        return self.data


class LabelledDataset(Dataset):
    def __init__(self: LabelledDataset, data: tf.Tensor):
        if tf.rank(data) != 2:
            raise ValueError("data must be a 2D tensor")
        if data.shape[1] < 2:
            raise ValueError("data must have at least 2 columns")
        elif data.shape[1] > 2:
            warn(
                "Only first, as data, and last, as labels, columns will be used",
                Warning,
            )
        super().__init__(data[:, 0])
        self.label: tf.Tensor = data[:, -1]

    def get_data(self: LabelledDataset) -> tf.data.Dataset:
        return self.data

    def get_label(self: LabelledDataset) -> tf.data.Dataset:
        return self.label


def load_txt(
    path: str, lines: bool = False, label: bool = False, extension: str = "txt"
) -> Dataset:
    path = Path(path)
    if label:
        if lines:
            return LabelledDataset(
                tf.convert_to_tensor(
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
                    ),
                    dtype=tf.string,
                )
            )
        else:  # label and not lines
            return LabelledDataset(
                tf.convert_to_tensor(
                    list(
                        map(
                            lambda f: (f.read_text(), f.parent.name),
                            path.glob("**/*." + extension),
                        )
                    ),
                    dtype=tf.string,
                )
            )
    else:  # not label
        if lines:
            return Dataset(
                tf.convert_to_tensor(
                    list(
                        chain.from_iterable(
                            map(
                                lambda f: f.read_text().splitlines(),
                                path.glob("**/*." + extension),
                            )
                        )
                    ),
                    dtype=tf.string,
                )
            )
        else:  # not label and not lines
            return Dataset(
                tf.convert_to_tensor(
                    list(map(lambda f: f.read_text(), path.glob("**/*." + extension))),
                    dtype=tf.string,
                )
            )
