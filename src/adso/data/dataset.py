"""Dataset class.

Define data-container for other classes.
"""

import json
import os
from pathlib import Path
from typing import Iterable, Tuple, Union

import dask.array as da
import numpy as np
from more_itertools import chuncked

from ..common import PROJDIR
from .corpus import Corpus, Raw


class Dataset:
    """Dataset class."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.path = PROJDIR / self.name
        # maybe add existence check?
        self.path.mkdir(exist_ok=True, parents=True)

        self.data: dict[str, Corpus] = {}

    def serialize(self) -> dict:
        save = {"name": self.name, "path": self.path}
        save["data"] = {key: self.data[key].serialize() for key in self.data}
        return save

    def save(self) -> None:
        with (self.path / (self.name + ".json")).open(mode="w") as f:
            json.dump(self.serialize(), f)

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "Dataset":
        path = Path(path)
        if path.is_dir():
            with (path / (path.name + ".json")).open(mode="r") as f:
                loaded = json.load(f)
        else:
            with path.open(mode="r") as f:
                loaded = json.load(f)
            path = path.parent

        dataset = cls(loaded["name"])
        dataset.path = path

        dataset.data = {
            key: globals()[loaded["data"][key]["format"]](
                Path(loaded["data"][key]["path"]), loaded["data"][key]["hash"]
            )
            for key in loaded["data"]
        }

        dataset.save()
        return dataset

    @classmethod
    def from_iterator(
        cls, name: str, iterator: Iterable[str], batch_size: int = 64
    ) -> "Dataset":
        dataset = cls(name)
        path = PROJDIR / name / (name + ".raw.hdf5")
        dataset.data["raw"] = Raw.from_iterator(path, iterator, batch_size)
        return dataset


class LabeledDataset(Dataset):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.label: dict[str, Corpus] = {}

    def serialize(self) -> dict:
        save = super().serialize()
        save["label"] = {key: self.label[key].serialize() for key in self.label}
        return save

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "LabeledDataset":

        path = Path(path)
        if path.is_dir():
            with (path / (path.name + ".json")).open(mode="r") as f:
                loaded = json.load(f)
        else:
            with path.open(mode="r") as f:
                loaded = json.load(f)
            path = path.parent

        dataset = cls(loaded["name"])
        dataset.path = path

        dataset.data = {
            key: globals()[loaded["data"][key]["format"]](
                Path(loaded["data"][key]["path"]), loaded["data"][key]["hash"]
            )
            for key in loaded["data"]
        }

        dataset.save()
        return dataset

    @classmethod
    def from_iterator(cls, name: str, iterator: Iterable[Tuple[str, str]], batch_size: int = 64) -> "LabeledDataset":  # type: ignore[override]
        dataset = cls(name)
        raw_path = PROJDIR / name / (name + ".raw.hdf5")
        label_path = PROJDIR / name / (name + ".label.raw.hdf5")
        data = da.concatenate(
            [da.from_array(np.array(chunk)) for chunk in chuncked(iterator, batch_size)]
        )
        dataset.data["raw"] = Raw.from_iterator(
            raw_path, data[:, 1].squeeze(), batch_size
        )
        dataset.label["raw"] = Raw.from_iterator(
            label_path, data[:, 0].squeeze(), batch_size
        )
        return dataset
