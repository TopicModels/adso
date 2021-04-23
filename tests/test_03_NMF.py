import adso
import adso.corpora as corpora


def test_NMF():

    adso.set_adso_dir(".test")
    adso.set_project_name("test")
    adso.set_seed(8686)

    dataset = corpora.get_20newsgroups(
        "NMF_20news", categories=["sci.space", "rec.autos"]
    )


if __name__ == "__main__":
    test_NMF()
