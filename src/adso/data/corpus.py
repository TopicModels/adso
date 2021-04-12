"Corpus Class and subclasses"

from abc import ABC
from pathlib import Path
from typing import Any

import dask.array as da
import h5py


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
