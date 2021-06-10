"Corpus Class and subclasses"

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Any, Optional

import dask.array as da
import dill
import numpy as np
import sparse as sp
import zarr

from ..common import Data, compute_hash
from .common import save_array_to_zarr


class Corpus(Data, ABC):
    pass


class Raw(Corpus):
    def get(self, skip_hash_check: bool = False) -> da.array:
        if self.hash == compute_hash(self.path):
            return da.from_zarr(zarr.open(store=zarr.ZipStore(self.path), mode="r"))
        else:
            raise RuntimeError("Different hash")

    @classmethod
    def from_dask_array(
        cls, path: Path, array: da.array, overwrite: bool = False
    ) -> Raw:
        if path.is_file() and (not overwrite):
            raise RuntimeError("File already exists")
        else:
            save_array_to_zarr(array, path)
        return cls(path)


class File(Corpus):
    def get(self, skip_hash_check: bool = False) -> Path:
        if self.hash == compute_hash(self.path):
            return self.path
        else:
            raise RuntimeError("Different hash")


class Pickled(Corpus):
    def get(self, skip_hash_check: bool = False) -> Any:
        if self.hash == compute_hash(self.path):
            return dill.load(self.path.open("rb"))
        else:
            raise RuntimeError("Different hash")

    @classmethod
    def from_object(cls, path: Path, obj: Any, overwrite: bool = False) -> Pickled:
        if path.is_file() and (not overwrite):
            raise RuntimeError("File already exists")
        else:
            dill.dump(obj, path.open("xb"))
        return cls(path)


class WithVocab(Corpus):
    def get(self, skip_hash_check: bool = False) -> da.array:
        if self.hash == compute_hash(self.path):
            return da.from_zarr(
                zarr.open(store=zarr.ZipStore(self.path), mode="r"), component="data"
            )
        else:
            raise RuntimeError("Different hash")

    def get_vocab(self, skip_hash_check: bool = False) -> da.array:
        if self.hash == compute_hash(self.path):
            return da.from_zarr(
                zarr.open(store=zarr.ZipStore(self.path), mode="r"), component="vocab"
            )
        else:
            raise RuntimeError("Different hash")

    @classmethod
    def from_dask_array(
        cls,
        path: Path,
        data: da.array,
        vocab: Optional[da.array],
        overwrite: bool = False,
    ) -> WithVocab:
        if path.is_file() and (not overwrite):
            raise RuntimeError("File already exists")
        else:
            if vocab is not None:
                save_array_to_zarr({"data": data, "vocab": vocab}, path)
            else:
                save_array_to_zarr({"data": data}, path)
        return cls(path)


class SparseWithVocab(WithVocab):
    def get(self, skip_hash_check: bool = False, sparse: bool = True) -> da.array:
        if sparse:
            return super().get().map_blocks(lambda x: sp.COO(x, fill_value=0))
        else:
            return super().get()


class Sparse(Raw):
    def get(self, skip_hash_check: bool = False, sparse: bool = True) -> da.array:
        if sparse:
            return super().get().map_blocks(lambda x: sp.COO(x, fill_value=0))
        else:
            return super().get()
