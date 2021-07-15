import adso
from adso.algorithms import hSBM
from adso.corpora import get_20newsgroups
from adso.metrics.supervised import NMI


def test_hSBM():

    adso.set_adso_dir(".test")
    adso.set_project_name("test")
    adso.set_seed(8686)

    dataset = get_20newsgroups("hSBM_20news", categories=["sci.space", "rec.autos"])

    hsbm = hSBM()

    topic_model, (n_layer,) = hsbm.fit_transform(dataset, "test_hSBM")

    assert round(NMI(dataset, topic_model), 5) == 0.1668


if __name__ == "__main__":
    import shutil

    try:
        shutil.rmtree(".test/test")
    except FileNotFoundError:
        pass

    test_hSBM()
