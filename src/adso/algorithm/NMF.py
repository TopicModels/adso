from pathlib import Path
import pickle
from typing import TYPE_CHECKING

import dask.array as da
import sklearn.decomposition

from ..common import PROJDIR, get_seed, compute_hash
from .common import TMAlgorithm, TopicModel

if TYPE_CHECKING:
    from ..data.dataset import Dataset


class NMF(TMAlgorithm):
    def __init__(self, name: str, n: int, overwrite: bool = False, **kwargs) -> None:
        path = PROJDIR / (name + ".pickle")
        super().__init__(path, name)
        model = sklearn.decomposition.NMF(
            n_components=n, random_state=get_seed(), **kwargs
        )
        if path.is_file() and (not overwrite):
            raise RuntimeError("File already exists")
        else:
            self.save(model)

    def save(self, model: sklearn.decomposition.NMF) -> None:  # type: ignore[override]
        pickle.dump(
            model,
            self.path.open("wb"),
        )
        self.update_hash()

    def get(self) -> sklearn.decomposition.NMF:
        if self.hash == compute_hash(self.path):
            return pickle.load(self.path.open("rb"))
        else:
            raise RuntimeError("Different hash")

    def fit_transform(self, dataset: "Dataset", path: Path, update: bool = True) -> TopicModel:  # type: ignore[override]

        model = self.get()

        doc_topic_matrix = da.from_array(
            model.fit_transform(dataset.get_frequency_matrix())
        )
        word_topic_matrix = da.from_array(model.components_).T

        topic_model = TopicModel.from_dask_array(
            path, word_topic_matrix, doc_topic_matrix
        )

        if update:
            self.save(model)

        return topic_model
