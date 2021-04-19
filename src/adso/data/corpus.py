"Corpus Class and subclasses"

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import dask.array as da
import h5py


from .common import hash


class Corpus(ABC):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.hash: str = hash(self.path)

    @abstractmethod
    def get(self) -> Any:
        raise NotImplementedError

    def serialize(self) -> Dict[str, str]:
        return {
            "format": type(self).__name__,
            "path": str(self.path),
            "hash": self.hash,
        }

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
            return da.from_array(h5py.File(self.path, "r")["/raw"])
        else:
            raise RuntimeError

    @classmethod
    def from_dask_array(cls, path: Path, array: da.array) -> "Raw":
        if path.is_file():
            raise RuntimeError
        else:
            array.to_hdf5(path, "/raw", shuffle=False)
        return Raw(path)
