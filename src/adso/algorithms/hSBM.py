from __future__ import annotations

from sys import modules
from typing import TYPE_CHECKING, Tuple

from ..data.topicmodel import HierarchicalTopicModel
from .common import TMAlgorithm
from .. import common

if TYPE_CHECKING:
    from ..data import Dataset

try:
    from ._sbmtm import sbmtm
except ImportError as e:
    print(e)
    print(
        "graph-tool not found, hSBM algorithm not available.\nInstall it with conda (graph-tool package) should resolve the issue"
    )


class hSBM(TMAlgorithm):
    # https://amaral.northwestern.edu/resources/software/topic-mapping
    # https://bitbucket.org/andrealanci/topicmapping/src/master/ReadMe.pdf
    if "graph_tool" in modules:

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def fit_transform(
            self,
            dataset: Dataset,
            name: str,
        ) -> Tuple[HierarchicalTopicModel, Tuple[int]]:
            model = sbmtm()
            print("Load")
            model.load_graph(str(dataset.get_gt_graph_path()))
            print("Fit")
            model.fit(**self.kwargs)
            # probabilmente la soluzione migliore è salvare il model, e scrivere hTM class per cachare le query (almento quelle semplici)
            print("Save model")
            n_layers: int = model.L
            hSBMDIR = common.PROJDIR / "hSBM" / name
            hSBMDIR.mkdir(exist_ok=True, parents=True)
            model.dump_model(filename=str(hSBMDIR / "model.pkl"))
            print("Save TM")
            return (
                HierarchicalTopicModel.from_array(
                    name,
                    [
                        (
                            model.get_groups(l=i)["p_w_tw"].T,
                            model.get_groups(l=i)["p_tw_d"].T,
                        )
                        for i in range(n_layers)
                    ],
                ),
                (n_layers,),
            )
