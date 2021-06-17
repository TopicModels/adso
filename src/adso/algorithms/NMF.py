from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import sklearn.decomposition
import sparse

from ..common import get_seed
from ..data.topicmodel import TopicModel
from .common import TMAlgorithm

if TYPE_CHECKING:
    from ..data.dataset import Dataset


class NMF(TMAlgorithm):
    def __init__(
        self, n: int, init: str = "nndsvd", max_iter: int = 1000, **kwargs
    ) -> None:
        self.model = sklearn.decomposition.NMF(
            n_components=n,
            random_state=get_seed(),
            init=init,
            max_iter=max_iter,
            **kwargs
        )

    def fit_transform(
        self, dataset: Dataset, name: str
    ) -> Tuple[TopicModel, Tuple[int, float]]:

        doc_topic_matrix = self.model.fit_transform(
            dataset.get_frequency_matrix().tocsr()
        )
        topic_word_matrix = self.model.components_

        topic_model = TopicModel.from_array(name, topic_word_matrix, doc_topic_matrix)

        return topic_model, (self.model.n_iter_, self.model.reconstruction_err_)
