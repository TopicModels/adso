import adso
import adso.data as data

adso.set_adso_dir(".test")
adso.set_project_name("test")


def test_from_iterator():

    labels = ["Animals", "Animals", "Maths", "Maths"]

    docs = [
        "Dinosaurs are reptiles. Birds descend from dinosaurs. Even if most of the dinosaurs don't fly, probably the majority of them was covered by feathers.",
        "Birds lay eggs, like reptiles and fishes. Most of the birds fly, even if penguins, ostriches and some others are unable to fly. Birds have two wings, like some dinosaurs and others ancient reptiles.",
        "Geometry studies shapes and entities in the space. A geometrician proves theorem about the relation among two or more geometrical entities. Continuity is a geometric concept widely used in calculus.",
        "Linear algebra studies matrices, vectors and vectorial spaces. Sometimes linear algebra is considered a subfield of geometry. Many theorem regarding matrices exists but one of the most important is the spectral one.",
    ]

    data.Dataset.from_iterator("test_from_iterator", docs)
    assert (
        data.Dataset.load(".test/test/test_from_iterator").path
        == data.Dataset.load(
            ".test/test/test_from_iterator/test_from_iterator.json"
        ).path
    )

    assert [
        x.decode("utf-8")
        for x in list(
            data.Dataset.load(".test/test/test_from_iterator").get_corpus().compute()
        )
    ] == docs

    data.LabeledDataset.from_iterator("labeled_test_from_iterator", zip(labels, docs))
    assert (
        data.LabeledDataset.load(".test/test/labeled_test_from_iterator").path
        == data.LabeledDataset.load(
            ".test/test/labeled_test_from_iterator/labeled_test_from_iterator.json"
        ).path
    )

    assert [
        x.decode("utf-8")
        for x in list(
            data.LabeledDataset.load(".test/test/labeled_test_from_iterator")
            .get_labels()
            .compute()
        )
    ] == labels


if __name__ == "__main__":
    test_from_iterator()