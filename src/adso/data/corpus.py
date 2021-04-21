"Corpus Class and subclasses"

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import dask.array as da
import h5py
import sparse

from .common import compute_hash


class Corpus(ABC):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.hash: Union[str, List[str]]
        self.update_hash()

    @abstractmethod
    def get(self) -> Any:
        raise NotImplementedError

    def update_hash(self) -> None:
        self.hash = compute_hash(self.path)

    def serialize(self) -> Dict[str, Union[str, List[str]]]:
        return {
            "format": type(self).__name__,
            "path": str(self.path),
            "hash": self.hash,
        }

    @classmethod
    def load(cls, path: Union[Path, str], hash: Optional[str]) -> "Corpus":
        path = Path(path)
        if path.is_file():
            corpus = cls(path)
            if (corpus.hash == hash) or (hash is None):
                return corpus
            else:
                raise RuntimeError("Different hash")
        else:
            raise RuntimeError("File doesn't exists")


class Raw(Corpus):
    def get(self, skip_hash_check: bool = False) -> da.array:
        if self.hash == compute_hash(self.path):
            return da.from_array(h5py.File(self.path, "r")["/raw"])
        else:
            raise RuntimeError("Different hash")

    @classmethod
    def from_dask_array(
        cls, path: Path, array: da.array, overwrite: bool = False
    ) -> "Raw":
        if path.is_file() and (not overwrite):
            raise RuntimeError("File already exists")
        else:
            array.to_hdf5(path, "/raw", shuffle=False)
        return Raw(path)


class CountMatrix(Corpus):
    def get(self, skip_hash_check: bool = False) -> da.array:
        if self.hash == compute_hash(self.path):
            return da.from_array(h5py.File(self.path, "r")["/count_matrix"]).map_blocks(
                sparse.COO
            )
        else:
            raise RuntimeError("Different hash")

    def get_vocab(self, skip_hash_check: bool = False) -> da.array:
        if self.hash == compute_hash(self.path):
            return da.from_array(h5py.File(self.path, "r")["/vocab"])
        else:
            raise RuntimeError("Different hash")

    @classmethod
    def from_dask_array(
        cls,
        path: Path,
        count_matrix: da.array,
        vocab: Optional[da.array],
        overwrite: bool = False,
    ) -> "CountMatrix":
        if path.is_file() and (not overwrite):
            raise RuntimeError("File already exists")
        else:
            if vocab is not None:
                da.to_hdf5(
                    path,
                    {"/count_matrix": count_matrix, "/vocab": vocab},
                    shuffle=False,
                )
            else:
                count_matrix.to_hdf5(path, "/count_matrix", shuffle=False)
        return CountMatrix(path)
