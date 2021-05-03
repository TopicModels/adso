from typing import TYPE_CHECKING, Optional

import dask.array as da
import dill
import sklearn.decomposition

from .. import common
from ..common import compute_hash, get_seed
from ..data.topicmodel import TopicModel
from .common import TMAlgorithm

if TYPE_CHECKING:
    from ..data.dataset import Dataset


class NMF(TMAlgorithm):
    def __init__(
        self,
        name: str,
        n: int,
        overwrite: bool = False,
        random_state: Optional[int] = get_seed(),
        init: str = "nndsvd",
        max_iter: int = 1000,
        **kwargs
    ) -> None:
        self.name = name
        self.path = common.PROJDIR / (self.name + ".NMF.pickle")

        model = sklearn.decomposition.NMF(
            n_components=n,
            random_state=random_state,
            init=init,
            max_iter=max_iter,
            **kwargs
        )
        if self.path.is_file() and (not overwrite):
            raise RuntimeError("File already exists")
        else:
            self.save(model)

    def save(self, model: sklearn.decomposition.NMF) -> None:  # type: ignore[override]
        dill.dump(
            model,
            self.path.open("wb"),
        )
        self.update_hash()

    def get(self) -> sklearn.decomposition.NMF:
        if self.hash == compute_hash(self.path):
            return dill.load(self.path.open("rb"))
        else:
            raise RuntimeError("Different hash")

    def fit_transform(self, dataset: "Dataset", name: Optional[str] = None, update: bool = True) -> TopicModel:  # type: ignore[override]

        if name is None:
            name = dataset.name + "_" + self.name

        model = self.get()

        doc_topic_matrix = da.from_array(
            model.fit_transform(dataset.get_frequency_matrix().compute().tocsr())
        )
        word_topic_matrix = da.from_array(model.components_).T

        topic_model = TopicModel.from_dask_array(
            name, word_topic_matrix, doc_topic_matrix
        )

        if update:
            self.save(model)

        return topic_model
