import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import dask.array as da
import sklearn.decomposition

from .. import common
from ..common import compute_hash, get_seed
from ..data.topicmodel import TopicModel
from .common import TMAlgorithm

if TYPE_CHECKING:
    from ..data.dataset import Dataset


class NMF(TMAlgorithm):
    def __init__(self, name: str, n: int, overwrite: bool = False, **kwargs) -> None:
        self.name = name
        self.path = common.PROJDIR / (self.name + ".NMF.pickle")

        model = sklearn.decomposition.NMF(
            n_components=n,
            random_state=get_seed(),
            solver="mu",
            beta_loss="kullback-leibler",
            **kwargs
        )
        if self.path.is_file() and (not overwrite):
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

    def fit_transform(self, dataset: "Dataset", path: Optional[Path] = None, update: bool = True) -> TopicModel:  # type: ignore[override]

        if path is None:
            path = common.PROJDIR / (self.name + ".NMF.topicmodel.hdf5")

        model = self.get()

        doc_topic_matrix = da.from_array(
            model.fit_transform(dataset.get_frequency_matrix().compute().tocsr())
        )
        word_topic_matrix = da.from_array(model.components_).T

        topic_model = TopicModel.from_dask_array(
            path, word_topic_matrix, doc_topic_matrix
        )

        if update:
            self.save(model)

        return topic_model
