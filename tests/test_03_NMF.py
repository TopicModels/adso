import adso
from adso.corpora import get_20newsgroups
from adso.algorithms import NMF
from adso.metrics.supervised import NMI, confusion_matrix


def test_NMF():

    adso.set_adso_dir(".test")
    adso.set_project_name("test")
    adso.set_seed(8686)

    dataset = get_20newsgroups("NMF_20news", categories=["sci.space", "rec.autos"])

    nmf = NMF("test_NMF", 2)

    topic_model = nmf.fit_transform(dataset)

    print(NMI(dataset, topic_model))

    print(confusion_matrix(dataset, topic_model).todense())


if __name__ == "__main__":
    import shutil

    try:
        shutil.rmtree(".test/test")
    except FileNotFoundError:
        pass

    test_NMF()
