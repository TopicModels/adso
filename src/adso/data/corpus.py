"Corpus Class and subclasses"

from __future__ import annotations

import zipfile
from abc import ABC
from pathlib import Path
from typing import Any, Optional

import dask.array as da
import dill
import numpy as np
import zarr

from ..common import Data
from .common import save_array_to_zarr


class Corpus(Data, ABC):
    pass


class Raw(Corpus):
    def get(self, skip_hash_check: bool = False) -> zarr.array:
        if self.check_hash() or skip_hash_check:
            return zarr.open(store=zarr.ZipStore(self.path), mode="r")
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

    @classmethod
    def from_array(cls, path: Path, array: np.ndarray, overwrite: bool = False) -> Raw:
        if path.is_file() and (not overwrite):
            raise RuntimeError("File already exists")
        else:
            with zarr.ZipStore(path, zipfile.ZIP_DEFLATED) as store:
                zarr.save_array(store, array)
        return cls(path)


class File(Corpus):
    def get(self, skip_hash_check: bool = False) -> Path:
        if self.check_hash() or skip_hash_check:
            return self.path
        else:
            raise RuntimeError("Different hash")


class Pickled(Corpus):
    def get(self, skip_hash_check: bool = False) -> Any:
        if self.check_hash() or skip_hash_check:
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
    def get(self, skip_hash_check: bool = False) -> zarr.array:
        if self.check_hash() or skip_hash_check:
            return zarr.open(store=zarr.ZipStore(self.path), mode="r")["data"]
        else:
            raise RuntimeError("Different hash")

    def get_vocab(self, skip_hash_check: bool = False) -> zarr.array:
        if self.check_hash() or skip_hash_check:
            return zarr.open(store=zarr.ZipStore(self.path), mode="r")["vocab"]
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

    @classmethod
    def from_array(
        cls,
        path: Path,
        data: np.ndarray,
        vocab: Optional[np.ndarray],
        overwrite: bool = False,
    ) -> WithVocab:
        if path.is_file() and (not overwrite):
            raise RuntimeError("File already exists")
        else:
            with zarr.ZipStore(path, compression=zipfile.ZIP_DEFLATED) as store:
                if vocab is not None:
                    zarr.save_group(store, data=data, vocab=vocab)
                else:
                    zarr.save_group(store, data=data)
        return cls(path)
