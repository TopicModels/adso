from __future__ import annotations

import numpy as np
import umap
import hdbscan

class UMAP_HDBSCAN(TMAlgorithm):
    def __init__(self, u_args: dict, h_args: dict, **kwargs) -> None:
        self.u_args = u_args
        self.h_args = h_args
        self.kwargs = kwargs

    def fit_transform(self, dataset: Dataset, name: str) -> TopicModel:
        # https://umap-learn.readthedocs.io/en/latest/index.html
        model = umap.UMAP(**u_args).fit(dataset.get_count_matrix())
        disc = umap.utils.disconnected_vertices(mapper)
        embedding = mapper.embedding_[~disc,:]
        
        # https://hdbscan.readthedocs.io/en/latest/index.html
        clusterer = hdbscan.HDBSCAN(**h_args).fit(embedding)
        #labels = clusterer.labels_
        doc_topic_matrix = hdbscan.all_points_membership_vectors(clusterer)        
        topic_word_matrix = np.array()
        
        return TopicModel.from_array(name, topic_word_matrix, doc_topic_matrix)
