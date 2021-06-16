from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import gensim.models.hdpmodel
import numpy as np
import tomotopy

from ..common import get_seed
from ..data.topicmodel import TopicModel
from .common import TMAlgorithm

if TYPE_CHECKING:
    from ..data import Dataset


class HDPVB(TMAlgorithm):
    # https://radimrehurek.com/gensim/models/hdpmodel.html
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def fit_transform(
        self, dataset: Dataset, name: str
    ) -> Tuple[TopicModel, Tuple[int]]:
        model = gensim.models.hdpmodel.HdpModel(
            dataset.get_gensim_corpus(),
            dataset.get_gensim_vocab(),
            random_state=get_seed(),
            **self.kwargs,
        )
        # To get back DT matrix https://github.com/bmabey/pyLDAvis/blob/master/pyLDAvis/gensim_models.py
        n = len(model.lda_alpha)
        topic_word_matrix = model.lda_beta
        doc_topic_matrix = model.inference(dataset.get_gensim_corpus())

        self.model = model
        return TopicModel.from_array(name, topic_word_matrix, doc_topic_matrix), (n,)


class HDPGS(TMAlgorithm):
    def __init__(self, n_iter: int = 1000, **kwargs) -> None:
        self.kwargs = kwargs
        self.n_iter = n_iter

    def fit_transform(
        self, dataset: Dataset, name: str
    ) -> Tuple[TopicModel, Tuple[int]]:
        # https://bab2min.github.io/tomotopy/v0.12.0/en/index.html#tomotopy.HDPModel
        model = tomotopy.HDPModel(
            corpus=dataset.get_tomotopy_corpus(),
            seed=get_seed(),
            **self.kwargs,
        )
        model.train(iter=self.n_iter)
        n_topic = model.k

        unordered_word_topic = np.array(
            [model.get_topic_word_dist(i) for i in range(n_topic)]
        )
        word_idx = [int(i) for i in model.vocabs]
        topic_word_matrix = np.zeros(shape=(n_topic, dataset.n_word()), dtype=np.float_)
        topic_word_matrix[:, word_idx] = unordered_word_topic

        doc_topic_matrix = np.array(
            [model.infer(d)[0] for d in model.docs], dtype=np.float_
        )

        self.model = model
        return TopicModel.from_array(name, topic_word_matrix, doc_topic_matrix), (
            n_topic,
        )
