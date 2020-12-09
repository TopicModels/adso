"""Datasets and datased related functions."""

from __future__ import annotations

from . import newsgroups, test
from .common import DATADIR, load_txt
from .dataset import Dataset, LabelledDataset
from .newsgroups import load_20newsgroups
from .test import load_labelled_test_dataset, load_test_dataset

# Create data folder if it not exists
DATADIR.mkdir(exist_ok=True, parents=True)
