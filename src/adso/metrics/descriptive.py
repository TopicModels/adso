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
    vocab = dataset.get_vocab()
    topic_word_matrix = model.get_topic_word_matrix()
    return [
        [vocab[idx] for idx in np.argsort(-row.squeeze())[:n]]
        for row in topic_word_matrix.islice()
    ]
