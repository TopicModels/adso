import adso
import adso.data as data

adso.set_adso_dir(".test")
adso.set_project_name("test")


def test_vectorizer():

    docs = [
        "Dinosaurs are reptiles. Birds descend from dinosaurs. Even if most of the dinosaurs don't fly, probably the majority of them was covered by feathers.",
        "Birds lay eggs, like reptiles and fishes. Most of the birds fly, even if penguins, ostriches and some others are unable to fly. Birds have two wings, like some dinosaurs and others ancient reptiles.",
        "Geometry studies shapes and entities in the space. A geometrician proves theorem about the relation among two or more geometrical entities. Continuity is a geometric concept widely used in calculus.",
        "Linear algebra studies matrices, vectors and vectorial spaces. Sometimes linear algebra is considered a subfield of geometry. Many theorem regarding matrices exists but one of the most important is the spectral one.",
    ]

    dataset = data.Dataset.from_iterator("test_from_iterator", docs, overwrite=True)

    dataset.set_vectorizer_params(stop_words=data.common.get_nltk_stopwords())

    count_matrix = dataset.get_count_matrix()
    vocab = dataset.get_vocab()

    print(count_matrix.compute())
    print(vocab.compute())


if __name__ == "__main__":
    test_vectorizer()