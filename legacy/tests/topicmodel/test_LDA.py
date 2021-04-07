"""Test load_txt function from data module."""

from adso.transform import Tokenizer, CountVectorizer
from adso.topicmodel import LDA
from adso.data.test import load_test_dataset


def test_LDA():
    data = load_test_dataset()

    tokenizer = Tokenizer()

    tok = tokenizer.transform(data)

    vect = CountVectorizer()

    m = vect.fit_transform(tok)

    lda = LDA(2, tolerance=0.001)

    lda.fit(m)


if __name__ == "__main__":
    test_LDA()
