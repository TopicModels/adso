"""Dataset class.

Define data-container for other classes.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Union

import dask.array as da
import numpy as np
from more_itertools import chunked

from .. import common as adso_common
from .corpus import Corpus, Raw


class Dataset:
    """Dataset class."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.path = adso_common.PROJDIR / self.name
        # maybe add existence check?
        self.path.mkdir(exist_ok=True, parents=True)

        self.data: Dict[str, Corpus] = {}

    def serialize(self) -> dict:
        save: Dict[str, Any] = {"name": self.name, "path": str(self.path)}
        save["data"] = {key: self.data[key].serialize() for key in self.data}
        return save

    def save(self) -> None:
        with (self.path / (self.name + ".json")).open(mode="w") as f:
            json.dump(self.serialize(), f, indent=4)

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
            key: globals()[loaded["data"][key]["format"]].load(
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
        path = adso_common.PROJDIR / name / (name + ".raw.hdf5")

        if path.is_file():
            raise RuntimeError
        else:
            da.concatenate(
                [
                    da.from_array(np.array(chunk, dtype=np.dtype(bytes)))
                    for chunk in chunked(iterator, batch_size)
                ]
            ).to_hdf5(path, "/raw", shuffle=False)

        dataset.data["raw"] = Raw(path)
        dataset.save()
        return dataset

    def get_corpus(self) -> da.array:
        return self.data["raw"].get()


class LabeledDataset(Dataset):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.labels: Dict[str, Corpus] = {}

    def serialize(self) -> dict:
        save = super().serialize()
        save["labels"] = {key: self.labels[key].serialize() for key in self.labels}
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
            key: globals()[loaded["data"][key]["format"]].load(
                Path(loaded["data"][key]["path"]), loaded["data"][key]["hash"]
            )
            for key in loaded["data"]
        }

        dataset.labels = {
            key: globals()[loaded["labels"][key]["format"]].load(
                Path(loaded["labels"][key]["path"]), loaded["labels"][key]["hash"]
            )
            for key in loaded["labels"]
        }

        dataset.save()
        return dataset

    @classmethod
    def from_iterator(cls, name: str, iterator: Iterable[Tuple[str, str]], batch_size: int = 64) -> "LabeledDataset":  # type: ignore[override]
        #
        # An alternative implementation can use itertool.tee + threading/async
        # https://stackoverflow.com/questions/50039223/how-to-execute-two-aggregate-functions-like-sum-concurrently-feeding-them-f
        # https://github.com/omnilib/aioitertools
        #
        # (Label, Doc)

        dataset = cls(name)
        data_path = adso_common.PROJDIR / name / (name + ".raw.hdf5")
        label_path = adso_common.PROJDIR / name / (name + ".label.raw.hdf5")
        data = da.concatenate(
            [
                da.from_array(np.array(chunk, dtype=np.dtype(bytes)))
                for chunk in chunked(iterator, batch_size)
            ]
        )

        if data_path.is_file():
            raise RuntimeError
        else:
            data[:, 1].squeeze().to_hdf5(data_path, "/raw", shuffle=False)
            dataset.data["raw"] = Raw(data_path)

        if label_path.is_file():
            raise RuntimeError
        else:
            data[:, 0].squeeze().to_hdf5(label_path, "/raw", shuffle=False)
            dataset.labels["raw"] = Raw(label_path)

        dataset.save()
        return dataset

    def get_labels(self) -> da.array:
        return self.labels["raw"].get()
