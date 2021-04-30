import dask.array as da
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sparse import COO

from ..data import LabeledDataset
from ..data.topicmodel import TopicModel


def NMI(dataset: LabeledDataset, model: TopicModel) -> float:
    return normalized_mutual_info_score(dataset.get_labels_vect(), model.get_labels())


def confusion_matrix(dataset: LabeledDataset, model: TopicModel) -> da.array:
    labels = dataset.get_labels_vect()
    predicted = model.get_labels()
    L = labels.shape[0]
    P = predicted.shape[0]
    if L != P:
        raise ValueError("Mismatching in document number")
    return COO.from_iter(
        (da.ones_like(labels, dtype=np.dtype("u1")), (labels, predicted)),
        shape=(
            da.unique(labels).compute().shape[0],
            da.unique(predicted).compute().shape[0],
        ),
        fill_value=0,
        dtype=np.dtype(int),
    )
