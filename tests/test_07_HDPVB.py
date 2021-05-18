import numpy as np

import adso
import adso.data as data
from adso.algorithms import HDPVB
from adso.corpora import get_20newsgroups
from adso.metrics.supervised import NMI, confusion_matrix


def test_simple_HDPVB():
    adso.set_adso_dir(".test")
    adso.set_project_name("test")
    adso.set_seed(8686)

    docs = ["A A B C D", "B B B A C", "E F E"]
    labels = ["1", "1", "2"]

    dataset = data.LabeledDataset.from_iterator("HDPVB_simple_data", zip(labels, docs))

    dataset.set_vectorizer_params(
        tokenizer=(lambda s: s.split(" ")),
    )

    hdp = HDPVB()

    topic_model, (n,) = hdp.fit_transform(dataset, "test_simple_HDPVB")

    assert round(NMI(dataset, topic_model), 5) == 0.27402
    assert (
        confusion_matrix(dataset, topic_model).todense() == np.array([[1, 1], [0, 1]])
    ).all()


def test_HDPVB():

    adso.set_adso_dir(".test")
    adso.set_project_name("test")
    adso.set_seed(8686)

    dataset = get_20newsgroups("HDPVB_20news", categories=["sci.space", "rec.autos"])

    hdp = HDPVB()

    topic_model, (n,) = hdp.fit_transform(dataset, "test_HDPVB")

    assert round(NMI(dataset, topic_model), 5) == 0.02955
    assert (
        confusion_matrix(dataset, topic_model).todense()
        == np.array(
            [
                [
                    618,
                    119,
                    95,
                    6,
                    72,
                    18,
                    10,
                    11,
                    4,
                    7,
                    7,
                    0,
                    3,
                    1,
                    0,
                    1,
                    2,
                    0,
                    0,
                    5,
                    2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                    1,
                    2,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                ],
                [
                    543,
                    174,
                    141,
                    34,
                    22,
                    10,
                    11,
                    14,
                    5,
                    6,
                    1,
                    3,
                    1,
                    4,
                    5,
                    1,
                    3,
                    2,
                    0,
                    0,
                    1,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ]
        )
    ).all()

    return dataset, topic_model


if __name__ == "__main__":
    import shutil

    try:
        shutil.rmtree(".test/test")
    except FileNotFoundError:
        pass

    test_simple_HDPVB()
    test_HDPVB()
