import numpy as np

import adso
import adso.data as data


def test_vectorizer():

    adso.set_adso_dir(".test")
    adso.set_project_name("test")

    docs = ["A A B C D", "B B B A C", "E F E"]

    dataset = data.Dataset.from_iterator("test_vectorizer", docs)

    dataset.set_vectorizer_params()

    count_matrix = dataset.get_count_matrix()
    vocab = dataset.get_vocab()

    assert (
        count_matrix.compute().todense()
        == np.array([[2, 1, 1, 1, 0, 0], [1, 3, 1, 0, 0, 0], [0, 0, 0, 0, 2, 1]])
    ).all()
    assert {w.decode() for w in vocab.compute()} == {"a", "b", "c", "d", "e", "f"}


if __name__ == "__main__":
    test_vectorizer()
