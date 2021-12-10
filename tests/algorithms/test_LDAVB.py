import adso
import adso.data as data
from adso.algorithms import LDAVB
from adso.corpora import get_20newsgroups


def test_simple_LDAVB():
    adso.set_adso_dir(".test")
    adso.set_project_name("test")
    adso.set_seed(8686)

    docs = ["A A B C D", "B B B A C", "E F E"]
    labels = ["1", "1", "2"]

    dataset = data.LabeledDataset.from_iterator("LDAVB_simple_data", zip(labels, docs))

    dataset.set_vectorizer_params(
        tokenizer=(lambda s: s.split(" ")),
    )

    lda = LDAVB(2)

    return lda.fit_transform(dataset, "test_simple_LDAVB")


def test_LDAVB():

    adso.set_adso_dir(".test")
    adso.set_project_name("test")
    adso.set_seed(8686)

    dataset = get_20newsgroups("LDAVB_20news", categories=["sci.space", "rec.autos"])

    lda = LDAVB(2)

    topic_model = lda.fit_transform(dataset, "test_LDAVB")

    return dataset, topic_model


if __name__ == "__main__":
    import shutil

    try:
        shutil.rmtree(".test/test")
    except FileNotFoundError:
        pass

    test_simple_LDAVB()
    test_LDAVB()
