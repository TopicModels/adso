from __future__ import annotations

import numpy as np
import sklearn.datasets

from .. import common
from ..common import get_seed
from ..data import LabeledDataset


def get_20newsgroups(name: str, overwrite: bool = False, **kwargs) -> LabeledDataset:
    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html
    SKDIR = common.DATADIR / "sklearn"
    SKDIR.mkdir(exist_ok=True, parents=True)
    bunch = sklearn.datasets.fetch_20newsgroups(
        data_home=(SKDIR),
        subset="all",
        shuffle=False,
        random_state=get_seed(),
        **kwargs
    )
    data = bunch.data
    labels = np.array([bunch.target_names[label] for label in bunch.target])
    return LabeledDataset.from_array(
        name,
        labels,
        data,
        overwrite=overwrite,
    )
