import adso
import adso.data as data
from adso.algorithms import HDPGS
from adso.corpora import get_20newsgroups
from adso.metrics.supervised import NMI


def test_simple_HDPGS():
    adso.set_adso_dir(".test")
    adso.set_project_name("test")
    adso.set_seed(8686)

    docs = ["A A B C D", "B B B A C", "E F E"]
    labels = ["1", "1", "2"]

    dataset = data.LabeledDataset.from_iterator("HDPGS_simple_data", zip(labels, docs))

    dataset.set_vectorizer_params(
        tokenizer=(lambda s: s.split(" ")),
    )

    hdp = HDPGS()

    topic_model, (n,) = hdp.fit_transform(dataset, "test_simple_HDPGS")
    assert round(NMI(dataset, topic_model), 5) == 1.0


def test_HDPGS():

    adso.set_adso_dir(".test")
    adso.set_project_name("test")
    adso.set_seed(8686)

    dataset = get_20newsgroups("HDPGS_20news", categories=["sci.space", "rec.autos"])

    hdp = HDPGS()

    topic_model, (n,) = hdp.fit_transform(dataset, "test_HDPGS")
    assert round(NMI(dataset, topic_model), 5) == 0.76613


if __name__ == "__main__":
    import shutil

    try:
        shutil.rmtree(".test/test")
    except FileNotFoundError:
        pass

    test_simple_HDPGS()
    test_HDPGS()
