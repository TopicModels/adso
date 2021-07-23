from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import dask.array as da
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sparse import COO

if TYPE_CHECKING:
    from ..data import LabeledDataset
    from ..data.topicmodel import TopicModel


def NMI(dataset: LabeledDataset, model: TopicModel, soft: bool = False) -> float:
    if soft:
        return normalized_mutual_info_score(
            dataset.get_labels_vect(), model.get_doc_topic_matrix(normalize=True)
        )
    else:
        return normalized_mutual_info_score(
            dataset.get_labels_vect(), model.get_labels()
        )


def confusion_matrix(
    dataset: LabeledDataset, model: TopicModel, n_topic: Optional[int] = None
) -> da.array:
    labels = dataset.get_labels_vect()
    predicted = model.get_labels()
    L = labels.shape[0]
    P = predicted.shape[0]
    if L != P:
        raise ValueError("Mismatching in document number")
    if n_topic:
        shape = (np.max(labels) + 1, n_topic + 1)
    else:
        shape = (
            np.max(labels) + 1,
            np.max(predicted) + 1,
        )
    return COO.from_iter(
        (np.ones_like(labels, dtype=np.dtype("u1")), (labels, predicted)),
        shape=shape,
        fill_value=0,
        dtype=np.dtype(int),
    )
