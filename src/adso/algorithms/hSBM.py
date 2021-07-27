from __future__ import annotations

from sys import modules
from typing import TYPE_CHECKING, Tuple

import dask.array as da
import dill
import numpy as np
import sparse

from .. import common
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
            dill.dump(model, (hSBMDIR / "model.pkl").open("wb"))
            print("Save TM")

            def get_groups(model: "sbmtm", l: int = 0) -> Tuple[da.array, da.array]:
                # rewrite from _sbmtm to use dask
                V = model.get_V()
                D = model.get_D()

                g = model.g
                state = model.state
                state_l = state.project_level(l).copy(overlap=True)
                state_l_edges = state_l.get_edge_blocks()  # labeled half-edges

                # count labeled half-edges, group-memberships
                B = state_l.get_B()

                id_dbw = np.zeros(g.edge_index_range, dtype=np.dtype(int))
                id_wb = np.zeros(g.edge_index_range, dtype=np.dtype(int))
                id_b = np.zeros(g.edge_index_range, dtype=np.dtype(int))
                weig = np.zeros(g.edge_index_range, dtype=np.dtype(int))

                for i, e in enumerate(g.edges()):
                    _, id_b[i] = state_l_edges[e]
                    id_dbw[i] = int(e.source())
                    id_wb[i] = int(e.target()) - D
                    weig[i] = g.ep["count"][e]

                n_bw = sparse.COO(
                    [id_b, id_wb], weig, shape=(B, V), fill_value=0
                )  # number of half-edges incident on word-node w and labeled as word-group tw

                del id_wb

                n_dbw = sparse.COO(
                    [id_dbw, id_b], weig, shape=(D, B), fill_value=0
                )  # number of half-edges incident on document-node d and labeled as word-group td

                del weig
                del id_b
                del id_dbw

                ind_w = np.where(np.sum(n_bw, axis=1) > 0)[0]
                n_bw = n_bw[ind_w, :]
                del ind_w

                ind_w2 = np.where(np.sum(n_dbw, axis=0) > 0)[0]
                n_dbw = n_dbw[:, ind_w2]
                del ind_w2

                # topic-distribution for words P(t_w | w)
                p_w_tw = n_bw / np.sum(n_bw, axis=1).todense()[:, np.newaxis]

                # Mixture of word-groups into documetns P(d | t_w)
                p_tw_d = n_dbw / np.sum(n_dbw, axis=0).todense()[np.newaxis, :]

                return (
                    da.array(p_w_tw).map_blocks(
                        lambda b: b.todense(), dtype=np.dtype(float)
                    ),
                    da.array(p_tw_d).map_blocks(
                        lambda b: b.todense(), dtype=np.dtype(float)
                    ),
                )

            return (
                HierarchicalTopicModel.from_dask_array(
                    name,
                    map(lambda i: get_groups(model, l=i), range(n_layers)),
                ),
                (n_layers,),
            )
