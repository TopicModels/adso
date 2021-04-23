import adso
from adso.corpora import get_20newsgroups
from adso.algorithms import NMF


def test_NMF():

    adso.set_adso_dir(".test")
    adso.set_project_name("test")
    adso.set_seed(8686)

    dataset = get_20newsgroups(
        "NMF_20news", overwrite=True, categories=["sci.space", "rec.autos"]
    )

    nmf = NMF("test_NMF", 2, overwrite=True)

    topic_model = nmf.fit_transform(dataset)


if __name__ == "__main__":
    test_NMF()
