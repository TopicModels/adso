"""Test load_txt function from data module."""

from adso.data.test import (
    _write_test_dataset,
    load_labelled_test_dataset,
    load_test_dataset,
)


def test_write_dir():
    _write_test_dataset(force=True)


def test_load():
    assert len(load_test_dataset(lines=True).get_data()) == 12
    assert len(load_labelled_test_dataset(lines=True).get_data()) == 12
    assert len(load_labelled_test_dataset(lines=True).get_label()) == 12
    assert set(load_labelled_test_dataset(lines=True).get_label()) == set(
        ["linearalgebra", "geometry", "dinosaur", "bird"]
    )

    assert len(load_test_dataset().get_data()) == 4
    assert len(load_labelled_test_dataset().get_data()) == 4
    assert len(load_labelled_test_dataset().get_label()) == 4
    assert set(load_labelled_test_dataset().get_label()) == set(["maths", "animals"])
