import adso
import numpy as np
from adso.algorithms import NMF
from adso.corpora import get_20newsgroups
from adso.metrics.supervised import NMI, confusion_matrix


def test_NMF():

    adso.set_adso_dir(".test")
    adso.set_project_name("test")
    adso.set_seed(8686)

    dataset = get_20newsgroups("NMF_20news", categories=["sci.space", "rec.autos"])

    nmf = NMF("test_NMF", 2)

    topic_model = nmf.fit_transform(dataset)

    assert NMI(dataset, topic_model) == 0.0011914237458472512

    assert confusion_matrix(dataset, topic_model).todense() == np.array(
        [[617, 373], [653, 334]]
    )


if __name__ == "__main__":
    import shutil

    try:
        shutil.rmtree(".test/test")
    except FileNotFoundError:
        pass

    test_NMF()
