from __future__ import annotations

from typing import TYPE_CHECKING

import hdbscan
import numpy as np
import umap

from ..common import get_seed
from ..data.topicmodel import TopicModel
from .common import TMAlgorithm

if TYPE_CHECKING:
    from ..data import Dataset


class UMAP_HDBSCAN(TMAlgorithm):
    def __init__(self, u_args: dict = {}, h_args: dict = {}, **kwargs) -> None:
        self.u_args = u_args
        self.h_args = h_args
        self.kwargs = kwargs

    def fit_transform(
        self, dataset: Dataset, name: str, remove_disc: bool = True
    ) -> TopicModel:
        # WARNING: setting a seed for reproducibility make the algorithm run on a single core (-> slower)
        seed = None
        if get_seed():
            seed = get_seed()
        # https://umap-learn.readthedocs.io/en/latest/index.html
        mapper = umap.UMAP(random_state=seed, **self.u_args).fit(
            dataset.get_count_matrix()
        )
        # WARNING: some points might be disconnected (np.inf)
        if remove_disc:
            disc = umap.utils.disconnected_vertices(mapper)
            embedding = mapper.embedding_[~disc, :]
        else:
            embedding = mapper.embedding_

        # https://hdbscan.readthedocs.io/en/latest/index.html
        clusterer = hdbscan.HDBSCAN(prediction_data=True, **self.h_args).fit(embedding)
        # labels = clusterer.labels_
        # predicted labels (hard clusters) with -1 for too noisy observations: how to return them?
        doc_topic_matrix = hdbscan.all_points_membership_vectors(clusterer)
        topic_word_matrix = np.array([])

        return TopicModel.from_array(name, topic_word_matrix, doc_topic_matrix)
