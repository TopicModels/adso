from typing import TYPE_CHECKING, Tuple

import dask.array as da
import numpy as np

from ..common import get_seed, xlogy
from ..data.topicmodel import TopicModel
from .common import TMAlgorithm

if TYPE_CHECKING:
    from ..data.dataset import Dataset


class PLSA(TMAlgorithm):
    def __init__(
        self,
        n: int,
        max_iter: int = 50,
        tol: float = 10e-4,
    ) -> None:
        self.n = n
        self.max_iter = max_iter
        self.tol = tol

    def fit_transform(
        self, dataset: "Dataset", name: str
    ) -> Tuple[TopicModel, int, float]:

        count = dataset.get_count_matrix(sparse=False)  # d,w

        # d, k  -- k -- k, w
        doc_topic, s, topic_word = da.linalg.svd_compressed(
            count, self.n, seed=get_seed(), compute=False
        )
        del s

        error = 0.0

        for _i in range(self.max_iter):
            # E step
            R = da.multiply(
                doc_topic[:, :, np.newaxis], topic_word[np.newaxis, :, :]
            )  # d, k, w
            R = R / R.sum(axis=1)[:, np.newaxis, :]
            # M step
            doc_topic = da.multiply(R, count[:, np.newaxis, :]).sum(axis=2)
            doc_topic = doc_topic / doc_topic.sum(axis=1)[:, np.newaxis]
            topic_word = R.sum(axis=0)
            topic_word = topic_word / topic_word.sum(axis=1)[:, np.newaxis]

            old_error = error
            error = xlogy(count, da.matmul(doc_topic, topic_word)).sum()
            if (old_error - error) < self.tol:
                break

        topic_model = TopicModel.from_dask_array(name, topic_word, doc_topic)

        return topic_model, _i, error.compute()  # type: ignore[attr-defined]
