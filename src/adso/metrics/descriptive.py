from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from ..data import LabeledDataset
    from ..data.topicmodel import TopicModel


# untested
def top_words(
    dataset: LabeledDataset, model: TopicModel, n: int = 10
) -> List[List[str]]:
    return [
        [dataset.get_vocab()[idx] for idx in np.argsort(-row.squeeze())[:n]]
        for row in model.get_doc_topic_matrix().islice()
    ]
