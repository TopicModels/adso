"""Datasets and datased related functions."""

from __future__ import annotations

from .common import DATADIR, load_txt
from .dataset import Dataset, LabelledDataset

from . import test

# Create data folder if it not exists
DATADIR.mkdir(exist_ok=True, parents=True)
