from typing import TYPE_CHECKING, Optional

import dask.array as da
import sklearn.decomposition

from ..common import get_seed
from ..data.topicmodel import TopicModel
from .common import TMAlgorithm

if TYPE_CHECKING:
    from ..data.dataset import Dataset


class NMF(TMAlgorithm):
    def __init__(
        self,
        n: int,
        random_state: Optional[int] = get_seed(),
        init: str = "nndsvd",
        max_iter: int = 1000,
        **kwargs
    ) -> None:
        self.model = sklearn.decomposition.NMF(
            n_components=n,
            random_state=random_state,
            init=init,
            max_iter=max_iter,
            **kwargs
        )

    def fit_transform(self, dataset: "Dataset", name: str) -> TopicModel:

        doc_topic_matrix = da.from_array(
            self.model.fit_transform(dataset.get_frequency_matrix().compute().tocsr())
        )
        topic_word_matrix = da.from_array(self.model.components_)

        topic_model = TopicModel.from_dask_array(
            name, topic_word_matrix, doc_topic_matrix
        )

        return topic_model
