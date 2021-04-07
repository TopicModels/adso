"""Test file for newsgroup module."""

import pytest

from adso.data import DATADIR, load_txt
from adso.data.newsgroups import load_20newsgroups


def test_20newsgroups():
    data = load_20newsgroups()
    assert len(data) == 15064
    assert len(set(data.get_labels())) == 20

    with pytest.raises(UnicodeDecodeError):
        load_txt(
            DATADIR / "20newsgroups" / "20news-bydate-train" / "rec.sport.baseball",
            label=True,
            extension=None,
            ignore_errors=False,
        )


if __name__ == "__main__":
    test_20newsgroups()
