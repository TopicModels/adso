from pathlib import Path
from typing import Tuple

import dask.array as da
import h5py

from ..common import Data, compute_hash


class TopicModel(Data):
    def get_word_topic_matrix(self, skip_hash_check: bool = False) -> da.array:
        if self.hash == compute_hash(self.path):
            return da.from_array(h5py.File(self.path, "r")["/word_topic"])
        else:
            raise RuntimeError("Different hash")

    def get_doc_topic_matrix(self, skip_hash_check: bool = False) -> da.array:
        if self.hash == compute_hash(self.path):
            return da.from_array(h5py.File(self.path, "r")["/doc_topic"])
        else:
            raise RuntimeError("Different hash")

    def get(self, skip_hash_check: bool = False) -> Tuple[da.array, da.array]:
        return (
            self.get_word_topic_matrix(),
            self.get_doc_topic_matrix(),
        )  # could give a concurrency error

    def get_labels(self) -> da.array:
        if self.hash != compute_hash(self.path):
            raise RuntimeError("Different hash")
        try:
            f = h5py.File(self.path, "a")  # work around for concurrent I/O
        except OSError:
            h5py.File(self.path).close()
            f = h5py.File(self.path, "a")

        if "/labels" in f:
            return da.from_array(f["/labels"])
        else:
            labels = da.argmax(da.from_array(f["/doc_topic"]), axis=1)
            dset = f.create_dataset(
                "/labels",
                shape=labels.shape,
                chunks=tuple([c[0] for c in labels.chunks]),  # from dask source to_hdf5
                dtype=labels.dtype,
            )
            da.store(labels, dset)
            f.close()
            self.update_hash()
            return da.from_array(h5py.File(self.path, "r")["/labels"])

    @classmethod
    def from_dask_array(
        cls,
        path: Path,
        word_topic_matrix: da.array,
        doc_topic_matrix: da.array,
        overwrite: bool = False,
    ) -> "TopicModel":
        if path.is_file() and (not overwrite):
            raise RuntimeError("File already exists")
        else:
            da.to_hdf5(
                path,
                {"/word_topic": word_topic_matrix, "/doc_topic": doc_topic_matrix},
                shuffle=False,
            )
        return TopicModel(path)
