"""Dataset class.

Define data-container for other classes.
"""

import json
import os
from pathlib import Path
from typing import Iterable, Union

from ..common import PROJDIR
from .corpus import Corpus  # , Raw


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
    def from_iterator(cls, iterator: Iterable[str]) -> "Dataset":
        pass


class LabeledDataset(Dataset):
    pass
