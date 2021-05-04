import numpy as np

import adso
import adso.data as data
from adso.algorithms import NMF
from adso.corpora import get_20newsgroups
from adso.metrics.supervised import NMI, confusion_matrix


def test_simple_NMF():
    adso.set_adso_dir(".test")
    adso.set_project_name("test")

    docs = ["A A B C D", "B B B A C", "E F E"]
    labels = ["1", "1", "2"]

    dataset = data.LabeledDataset.from_iterator("test_simple_NMF", zip(docs, labels))

    dataset.set_vectorizer_params(
        tokenizer=(lambda s: s.split(" ")),
    )

    nmf = NMF("test_simple_NMF", 2)

    topic_model = nmf.fit_transform(dataset)

    assert round(NMI(dataset, topic_model), 5) == 0.73368
    assert (
        confusion_matrix(dataset, topic_model).todense()
        == np.array([[1, 0], [1, 0], [0, 1]])
    ).all()


def test_NMF():

    adso.set_adso_dir(".test")
    adso.set_project_name("test")
    adso.set_seed(8686)

    dataset = get_20newsgroups("NMF_20news", categories=["sci.space", "rec.autos"])

    nmf = NMF("test_NMF", 2)

    topic_model = nmf.fit_transform(dataset)

    assert round(NMI(dataset, topic_model), 5) == 0.00119
    assert (
        confusion_matrix(dataset, topic_model).todense()
        == np.array([[617, 373], [653, 334]])
    ).all()


if __name__ == "__main__":
    import shutil

    try:
        shutil.rmtree(".test/test")
    except FileNotFoundError:
        pass

    test_simple_NMF()
    test_NMF()
