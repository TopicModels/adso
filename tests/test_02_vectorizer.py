import adso
import adso.data as data


def test_vectorizer():

    adso.set_adso_dir(".test")
    adso.set_project_name("test")

    docs = ["A A B C D", "B B B A C", "E F E"]

    dataset = data.Dataset.from_iterator("test_vectorizer", docs)

    dataset.set_vectorizer_params(stop_words=data.common.get_nltk_stopwords())

    # count_matrix = dataset.get_count_matrix()
    # vocab = dataset.get_vocab()

    # print(count_matrix.compute())
    # print(vocab.compute())


if __name__ == "__main__":
    test_vectorizer()