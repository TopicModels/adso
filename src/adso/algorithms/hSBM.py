from __future__ import annotations

from sys import modules
from typing import TYPE_CHECKING

import dask.array as da

from ..common import get_seed
from ..data.topicmodel import HierarchicalTopicModel
from .common import TMAlgorithm

if TYPE_CHECKING:
    from ..data import Dataset

try:
    from ._vendor.sbmtm import sbmtm
except ImportError:
    print(
        "graph-tool not found, hSBM algorithm not available.\nInstall it with conda (graph-tool package) should resolve the issue"
    )


class hSBM(TMAlgorithm):
    # https://amaral.northwestern.edu/resources/software/topic-mapping
    # https://bitbucket.org/andrealanci/topicmapping/src/master/ReadMe.pdf
    if "graph_tool" in modules:

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def fit_transform(self, dataset: Dataset, name: str) -> HierarchicalTopicModel:
            model = sbmtm()
            model.load_graph(str(dataset.get_gt_graph_path()))
            model.fit(**self.kwargs)
            return None
