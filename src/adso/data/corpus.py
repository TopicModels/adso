"Corpus Class and subclasses"

from abc import ABC
from pathlib import Path
from typing import Any, Iterable

import dask.array as da
import h5py
import numpy as np
from more_itertools import chuncked

from .common import hash


class Corpus(ABC):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.hash: str = hash(self.path)

    def get(self) -> Any:
        pass

    def serialize(self) -> dict:
        return {"format": type(self).__name__, "path": self.path, "hash": self.hash}

    @classmethod
    def load(cls, path: Path, hash: str) -> "Corpus":
        if path.is_file():
            corpus = cls(path)
            if corpus.hash == hash:
                return corpus
            else:
                raise RuntimeError
        else:
            raise RuntimeError


class Raw(Corpus):
    def get(self) -> da.array:
        if self.hash == hash(self.path):
            return da.from_array(h5py.File(self.path, "r")["/"], chuncks="auto")
        else:
            raise RuntimeError

    @classmethod
    def from_iterator(
        cls, path: Path, iterator: Iterable[Any], batch_size: int = 64
    ) -> "Raw":
        if path.is_file():
            raise RuntimeError
        else:
            da.concatenate(
                [
                    da.from_array(np.array(chunk))
                    for chunk in chuncked(iterator, batch_size)
                ]
            ).to_hdf5(path, "/", shuffle=False)
        return Raw(path)
