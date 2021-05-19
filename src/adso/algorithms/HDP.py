from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import dask.array as da
import gensim.models.hdpmodel
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
        topic_word_matrix = da.from_array(model.lda_beta)
        doc_topic_matrix = da.from_array(model.inference(dataset.get_gensim_corpus()))

        self.model = model
        return TopicModel.from_dask_array(name, topic_word_matrix, doc_topic_matrix), (
            n,
        )


class HDPGS(TMAlgorithm):
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def fit_transform(
        self, dataset: Dataset, name: str
    ) -> Tuple[TopicModel, Tuple[int]]:
        # https://bab2min.github.io/tomotopy/v0.12.0/en/index.html#tomotopy.HDPModel
        model = tomotopy.HDPModel(
            corpus=dataset.get_tomotopy_corpus(),
            seed=get_seed(),
            **self.kwargs,
        )
        n = model.k
        # topic_word_matrix
        # doc_topic_matrix

        self.model = model
        # return TopicModel.from_dask_array(name, topic_word_matrix, doc_topic_matrix), (
        #    n,
        # )
