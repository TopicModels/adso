import adso
import adso.data as data
from adso.algorithms import LDAGS
from adso.corpora import get_20newsgroups


def test_simple_LDAGS():
    adso.set_adso_dir(".test")
    adso.set_project_name("test")
    adso.set_seed(8686)

    docs = ["A A B C D", "B B B A C", "E F E"]
    labels = ["1", "1", "2"]

    dataset = data.LabeledDataset.from_iterator("LDAGS_simple_data", zip(labels, docs))

    dataset.set_vectorizer_params(
        tokenizer=(lambda s: s.split(" ")),
    )

    lda = LDAGS(
        2,
        memory="512M",
        mallet_args={
            "num-iterations": 1000,
            "optimize-interval": 20,
        },
    )

    lda.fit_transform(dataset, "test_simple_LDAGS")


def test_LDAGS():

    adso.set_adso_dir(".test")
    adso.set_project_name("test")
    adso.set_seed(8686)

    dataset = get_20newsgroups(
        "LDAGS_20news",
        categories=[
            "sci.space",
            "rec.autos",
        ],
    )

    lda = LDAGS(
        2,
        mallet_args={"optimize-interval": 20},
    )

    lda.fit_transform(dataset, "test_LDAGS")


if __name__ == "__main__":
    import shutil

    try:
        shutil.rmtree(".test/test")
    except FileNotFoundError:
        pass

    test_simple_LDAGS()
    test_LDAGS()
