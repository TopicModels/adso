import adso
from adso.corpora import get_20newsgroups
from adso.algorithms import NMF
from adso.metrics.supervised import NMI


def test_NMF():

    adso.set_adso_dir(".test")
    adso.set_project_name("test")
    adso.set_seed(8686)

    dataset = get_20newsgroups("NMF_20news", categories=["sci.space", "rec.autos"])

    nmf = NMF("test_NMF", 2)

    topic_model = nmf.fit_transform(dataset)

    print(NMI(dataset, topic_model))


if __name__ == "__main__":
    test_NMF()
