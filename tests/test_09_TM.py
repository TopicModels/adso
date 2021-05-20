import numpy as np

import adso
import adso.data as data
from adso.algorithms import TopicMapping
from adso.corpora import get_20newsgroups

from adso.metrics.supervised import NMI, confusion_matrix


def test_simple_TM():
    adso.set_adso_dir(".test")
    adso.set_project_name("test")
    adso.set_seed(8686)

    docs = ["A A B C D", "B B B A C", "E F E"]
    labels = ["1", "1", "2"]

    dataset = data.LabeledDataset.from_iterator("TM_simple_data", zip(labels, docs))

    dataset.set_vectorizer_params(
        tokenizer=(lambda s: s.split(" ")),
    )

    tm = TopicMapping(p=1)

    topic_model, (n,) = tm.fit_transform(dataset, "test_simple_TM")

    assert round(NMI(dataset, topic_model), 5) == 1.0
    assert (
        confusion_matrix(dataset, topic_model).todense() == np.array([[2, 0], [0, 1]])
    ).all()


def test_TM():

    adso.set_adso_dir(".test")
    adso.set_project_name("test")
    adso.set_seed(8686)

    dataset = get_20newsgroups("TM_20news", categories=["sci.space", "rec.autos"])

    tm = TopicMapping()

    topic_model, (n,) = tm.fit_transform(dataset, "test_TM")

    assert round(NMI(dataset, topic_model), 5) == 0.16266
    assert (
        confusion_matrix(dataset, topic_model).todense()
        == np.array(
            [
                [18, 12, 45, 7, 294, 21, 71, 26, 6, 28, 92, 115, 86, 81, 4, 84],
                [162, 105, 58, 73, 2, 39, 79, 52, 44, 81, 36, 53, 54, 6, 130, 13],
            ]
        )
    ).all()


if __name__ == "__main__":
    import shutil

    try:
        shutil.rmtree(".test/test")
    except FileNotFoundError:
        pass

    test_simple_TM()
    test_TM()
