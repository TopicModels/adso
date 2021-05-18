import numpy as np

import adso
import adso.data as data
from adso.algorithms import LDAGS
from adso.corpora import get_20newsgroups
from adso.metrics.supervised import NMI, confusion_matrix


def test_simple_LDAGS():
    adso.set_adso_dir(".test")
    adso.set_project_name("test")
    adso.set_seed(8686)

    docs = ["A A B C D", "B B B A C", "E F E"]
    labels = ["1", "1", "2"]

    dataset = data.LabeledDataset.from_iterator("LDAGS_simple_data", zip(labels, docs))

    dataset.set_vectorizer_params(
        tokenizer=(lambda s: s.split(" ")),
    )

    lda = LDAGS(
        2,
        memory="512M",
        mallet_args={
            "num-iterations": 1000,
            "optimize-interval": 20,
        },
    )

    topic_model = lda.fit_transform(dataset, "test_simple_LDAGS")

    assert round(NMI(dataset, topic_model), 5) == 1.0
    assert (
        confusion_matrix(dataset, topic_model).todense() == np.array([[0, 2], [1, 0]])
    ).all()


def test_LDAGS():

    adso.set_adso_dir(".test")
    adso.set_project_name("test")
    adso.set_seed(8686)

    dataset = get_20newsgroups(
        "LDAGS_20news",
        categories=[
            "sci.space",
            "rec.autos",
        ],
    )

    lda = LDAGS(
        2,
        mallet_args={"optimize-interval": 20},
    )

    topic_model = lda.fit_transform(dataset, "test_LDAGS")

    assert round(NMI(dataset, topic_model), 5) == 0.80648
    assert (
        confusion_matrix(dataset, topic_model).todense()
        == np.array([[11, 979], [935, 52]])
    ).all()

    return dataset, topic_model


if __name__ == "__main__":
    import shutil

    try:
        shutil.rmtree(".test/test")
    except FileNotFoundError:
        pass

    test_simple_LDAGS()
    test_LDAGS()
