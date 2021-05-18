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

    # assert (round(NMI(dataset, topic_model), 5) == 0.02955)


if __name__ == "__main__":
    import shutil

    try:
        shutil.rmtree(".test/test")
    except FileNotFoundError:
        pass

    test_simple_HDPVB()
    test_HDPVB()
