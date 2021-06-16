from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List

import dask.array as da
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sparse import COO

if TYPE_CHECKING:
    from ..data import LabeledDataset
    from ..data.topicmodel import TopicModel

# untested
def top_words(
    dataset: LabeledDataset, model: TopicModel, n: int = 10
) -> List[List[str]]:
    return [
        [dataset.get_vocab()[idx] for idx in np.argsort(-row.squeeze())[:n]]
        for row in model.get_doc_topic_matrix()
    ]
