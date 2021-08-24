import numpy as np

import adso
import adso.data as data
from adso.algorithms import UMAP_HDBSCAN
from adso.corpora import get_20newsgroups
from adso.metrics.supervised import NMI, confusion_matrix


def test_simple_UMAP_HDBSCAN():
    adso.set_adso_dir(".test")
    adso.set_project_name("test")
    adso.set_seed(8686)

    docs = ["A A B C D", "B B B A C", "E F E"]
    labels = ["1", "1", "2"]

    dataset = data.LabeledDataset.from_iterator("PLSA_simple_data", zip(labels, docs))

    dataset.set_vectorizer_params(
        tokenizer=(lambda s: s.split(" ")),
    )

    model = UMAP_HDBSCAN()

    topic_model = model.fit_transform(dataset, "test_simple_UMAP_HDBSCAN")

    assert round(NMI(dataset, topic_model), 5) == 1.0
    assert (
        confusion_matrix(dataset, topic_model).todense() == np.array([[2, 0], [0, 1]])
    ).all()


def test_UMAP_HDBSCAN():

    adso.set_adso_dir(".test")
    adso.set_project_name("test")
    adso.set_seed(8686)

    dataset = get_20newsgroups("PLSA_20news", categories=["sci.space", "rec.autos"])

    model = UMAP_HDBSCAN()

    topic_model = model.fit_transform(dataset, "test_simple_UMAP_HDBSCAN")

    assert round(NMI(dataset, topic_model), 5) == 0.00032
    assert (
        confusion_matrix(dataset, topic_model).todense()
        == np.array([[521, 469], [540, 447]])
    ).all()

    return dataset, topic_model


if __name__ == "__main__":
    import shutil

    try:
        shutil.rmtree(".test/test")
    except FileNotFoundError:
        pass

    test_simple_UMAP_HDBSCAN()
    test_UMAP_HDBSCAN()
