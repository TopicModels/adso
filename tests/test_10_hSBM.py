import numpy as np

import adso
import adso.data as data
from adso.algorithms import hSBM
from adso.corpora import get_20newsgroups

from adso.metrics.supervised import NMI, confusion_matrix


def test_hSBM():

    adso.set_adso_dir(".test")
    adso.set_project_name("test")
    adso.set_seed(8686)

    dataset = get_20newsgroups("hSBM_20news", categories=["sci.space", "rec.autos"])

    hsbm = hSBM()

    topic_model, (n,) = hsbm.fit_transform(dataset, "test_hSBM")

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

    test_hSBM()
