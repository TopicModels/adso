from __future__ import annotations

import numpy as np
import umap
import hdbscan

from .. import common
from ..common import get_seed
from ..data.topicmodel import TopicModel
from .common import TMAlgorithm

class UMAP_HDBSCAN(TMAlgorithm):
    def __init__(self, u_args: dict, h_args: dict, **kwargs) -> None:
        self.u_args = u_args
        self.h_args = h_args
        self.kwargs = kwargs

    def fit_transform(self, dataset: Dataset, name: str, remove_disc = True) -> TopicModel:
        # WARNING: setting a seed for reproducibility make the algorithm run on a single core (-> slower)
        seed = None
        if get_seed():
            seed = get_seed()
        # https://umap-learn.readthedocs.io/en/latest/index.html
        model = umap.UMAP(random_state = seed, **u_args).fit(dataset.get_count_matrix())
        # WARNING: some points might be disconnected (np.inf)
        if remove_disc:
            disc = umap.utils.disconnected_vertices(mapper)
            embedding = mapper.embedding_[~disc,:]
        else:
            embedding = mapper.embedding_
        
        # https://hdbscan.readthedocs.io/en/latest/index.html
        clusterer = hdbscan.HDBSCAN(prediction_data = True, **h_args).fit(embedding)
        #labels = clusterer.labels_
        doc_topic_matrix = hdbscan.all_points_membership_vectors(clusterer)        
        topic_word_matrix = np.array()
        
        return TopicModel.from_array(name, topic_word_matrix, doc_topic_matrix)
