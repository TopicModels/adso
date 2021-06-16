from __future__ import annotations

from sys import modules
from typing import TYPE_CHECKING, Any, Tuple

from ..data.topicmodel import HierarchicalTopicModel
from .common import TMAlgorithm

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
            self, dataset: Dataset, name: str
        ) -> Tuple[HierarchicalTopicModel, Tuple[int, Any]]:
            model = sbmtm()
            model.load_graph(str(dataset.get_gt_graph_path()))
            model.fit(**self.kwargs)
            # probabilmente la soluzione migliore è salvare il model, e scivere hTM class per cachare le query (almento quelle semplici)
            n_layers: int = model.L
            results = []
            for i in range(n_layers):
                res = model.get_groups(l=i)
                results.append(
                    {
                        "n_topic": res["Bw"],
                        "n_docgroup": res["Bd"],
                        "doc_topic": res["p_tw_d"].T,
                        "topic_word": res["p_w_tw"].T,
                        "doc_docgroup": res["p_td_d"].T,
                    }
                )

            return (
                HierarchicalTopicModel.from_array(
                    name,
                    [
                        (
                            res["topic_word"],
                            res["doc_topic"],
                        )
                        for res in results
                    ],
                ),
                (n_layers, results),
            )
