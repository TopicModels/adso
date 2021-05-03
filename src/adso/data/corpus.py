"Corpus Class and subclasses"

from abc import ABC
from pathlib import Path
from typing import Optional

import dask.array as da
import h5py
import numpy as np
import sparse

from ..common import Data, compute_hash


class Corpus(Data, ABC):
    pass


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
            try:
                array.to_hdf5(path, "/raw", shuffle=False)
            except TypeError:
                if array.dtype.kind == "U":
                    itemsize = np.dtype("U1").itemsize
                elif array.dtype.kind == "S":
                    itemsize = np.dtype("S1").itemsize
                else:
                    raise TypeError("Numpy dtype not recognized")
                array.map_blocks(
                    lambda b: np.char.encode(b, encoding="utf-8"),
                    dtype=np.dtype(("S", array.itemsize // itemsize)),
                ).to_hdf5(path, "/raw", shuffle=False)
        return Raw(path)


class WithVocab(Corpus):
    def get(self, skip_hash_check: bool = False) -> da.array:
        if self.hash == compute_hash(self.path):
            return da.from_array(h5py.File(self.path, "r")["/data"])
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
        data: da.array,
        vocab: Optional[da.array],
        overwrite: bool = False,
    ) -> "WithVocab":
        if path.is_file() and (not overwrite):
            raise RuntimeError("File already exists")
        else:
            if vocab is not None:
                try:
                    da.to_hdf5(
                        path,
                        {"/data": data, "/vocab": vocab},
                        shuffle=False,
                    )
                except TypeError:
                    if vocab.dtype.kind == "U":
                        itemsize = np.dtype("U1").itemsize
                    elif vocab.dtype.kind == "S":
                        itemsize = np.dtype("S1").itemsize
                    else:
                        raise TypeError("Numpy dtype not recognized")
                    da.to_hdf5(
                        path,
                        {
                            "/data": data,
                            "/vocab": vocab.map_blocks(
                                lambda b: np.char.encode(b, encoding="utf-8"),
                                dtype=np.dtype(("S", vocab.itemsize // itemsize)),
                            ),
                        },
                        shuffle=False,
                    )
            else:
                data.to_hdf5(path, "/data", shuffle=False)
        return cls(path)


class SparseWithVocab(WithVocab):
    def get(self, skip_hash_check: bool = False) -> da.array:
        return super().get().map_blocks(lambda x: sparse.COO(x, fill_value=0))


class Sparse(Raw):
    def get(self, skip_hash_check: bool = False) -> da.array:
        return super().get().map_blocks(lambda x: sparse.COO(x, fill_value=0))
