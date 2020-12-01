from adso.data.newsgroups import load_20newsgroups


def test_20newsgroups():
    data = load_20newsgroups()
    assert len(data) == 15064
    assert len(set(data.get_labels())) == 20


test_20newsgroups()
