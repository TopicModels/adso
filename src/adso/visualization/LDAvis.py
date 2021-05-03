import json

import numpy as np
import pandas as pd
import pyLDAvis

from .. import common
from ..common import Data, compute_hash
from ..data import Dataset
from ..data.topicmodel import TopicModel


class Visualizer(Data):
    def display(self) -> None:
        pyLDAvis.display(self.get_vis())

    def show(self, **kwargs) -> None:
        pyLDAvis.show(self.get_vis(), **kwargs)

    def get_vis(self) -> pyLDAvis.PreparedData:
        # https://github.com/bmabey/pyLDAvis/issues/134
        if self.hash == compute_hash(self.path):
            vis_data = json.load(self.path.open())
            topic_coordinates = pd.DataFrame.from_dict(vis_data["mdsDat"])
            topic_info = pd.DataFrame.from_dict(vis_data["tinfo"])
            token_table = pd.DataFrame.from_dict(vis_data["token.table"])
            R = vis_data["R"]
            lambda_step = vis_data["lambda.step"]
            plot_opts = vis_data["plot.opts"]
            client_topic_order = vis_data["topic.order"]

            return pyLDAvis.PreparedData(
                topic_coordinates,
                topic_info,
                token_table,
                R,
                lambda_step,
                plot_opts,
                client_topic_order,
            )
        else:
            raise RuntimeError("Different hash")

    @classmethod
    def new(
        cls, name: str, dataset: Dataset, model: TopicModel, **kwargs
    ) -> "Visualizer":
        path = common.PROJDIR / (name + ".LDAvis.json")
        pyLDAvis.save_json(
            pyLDAvis.prepare(
                model.get_word_topic_matrix().T.compute(),
                model.get_doc_topic_matrix().compute(),
                dataset.get_count_matrix()
                .sum(axis=1)
                .map_blocks(lambda b: b.todense(), dtype=np.int64)
                .squeeze()
                .compute(),
                dataset.get_vocab().compute(),
                dataset.get_count_matrix()
                .sum(axis=0)
                .map_blocks(lambda b: b.todense(), dtype=np.int64)
                .squeeze()
                .compute(),
                **kwargs
            ),
            path,
        )
        return cls(path)
