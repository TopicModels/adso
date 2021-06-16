from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
from scipy.special import xlogy

from ..data.topicmodel import TopicModel
from .common import TMAlgorithm

if TYPE_CHECKING:
    from ..data.dataset import Dataset


class PLSA(TMAlgorithm):
    def __init__(
        self,
        n: int,
        max_iter: int = 50,
        tol: float = 10e-5,
    ) -> None:
        self.n = n
        self.max_iter = max_iter
        self.tol = tol

    def fit_transform(
        self, dataset: "Dataset", name: str
    ) -> Tuple[TopicModel, Tuple[int, float]]:

        count = dataset.get_count_matrix()[...]  # d,w

        # d, k  -- k -- k, w
        # doc_topic, s, topic_word = np.linalg.svd_compressed(
        #     count, self.n, seed=get_seed(), compute=False
        # )
        # del s

        if min(count.shape) > max(self.n, 5):
            doc_topic = np.stack(
                [
                    count[:, i].sum(axis=1)
                    for i in np.array_split(np.arange(count.shape[1]), self.n)
                ],
                axis=1,
            )
            topic_word = np.stack(
                [
                    count[i, :].sum(axis=0)
                    for i in np.array_split(np.arange(count.shape[0]), self.n)
                ],
                axis=0,
            )

        else:
            doc_topic = np.random.uniform(size=(count.shape[0], self.n))
            topic_word = np.random.uniform(size=(self.n, count.shape[1]))
        doc_topic = doc_topic / doc_topic.sum(axis=1)[:, np.newaxis]
        topic_word = topic_word / topic_word.sum(axis=1)[:, np.newaxis]

        error = 0.0

        for _i in range(self.max_iter):
            # E step
            R = np.multiply(
                doc_topic[:, :, np.newaxis], topic_word[np.newaxis, :, :]
            )  # d, k, w
            R = R / R.sum(axis=1)[:, np.newaxis, :]
            # M step
            doc_topic = np.multiply(R, count[:, np.newaxis, :]).sum(axis=2)
            doc_topic = doc_topic / doc_topic.sum(axis=1)[:, np.newaxis]
            topic_word = R.sum(axis=0)
            topic_word = topic_word / topic_word.sum(axis=1)[:, np.newaxis]

            old_error = error
            error = xlogy(count, np.matmul(doc_topic, topic_word)).sum()
            if (old_error - error) < self.tol:
                break

        topic_model = TopicModel.from_array(name, topic_word, doc_topic)

        return topic_model, (_i, error.compute())  # type: ignore[attr-defined]
