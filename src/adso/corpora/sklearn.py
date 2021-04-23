import numpy as np
import sklearn.dataset

from ..common import DATADIR, get_seed
from ..data import LabeledDataset


def get_20newsgroups(name: str, **kwargs) -> LabeledDataset:
    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html
    data, labels = sklearn.dataset.fetch_20newsgroups(
        data_home=(DATADIR / "sklearn"),
        subset="all",
        shuffle=False,
        random_state=get_seed(),
        return_X_y=True,
        **kwargs
    )

    return LabeledDataset.from_iterator(name, np.column_stack(labels, data))
