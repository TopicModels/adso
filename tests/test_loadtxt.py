"""Test load_txt function from data module."""

from adso.data.Test import (_write_test_dataset, load_labelled_test_dataset,
                            load_test_dataset)
import tensorflow as tf


def test_write_dir():
    _write_test_dataset(force=True)


def test_load():
    assert load_test_dataset(lines=True).get_data().shape == (12,)
    assert load_labelled_test_dataset(lines=True).get_data().shape == (12,)
    assert load_labelled_test_dataset(lines=True).get_label().shape == (12,)
    assert tf.reduce_all(
        tf.equal(
            tf.unique(load_labelled_test_dataset(lines=True).get_label())[0],
            tf.convert_to_tensor(["linearalgebra", "geometry", "dinosaur", "bird"]),
        )
    )

    assert load_test_dataset().get_data().shape == (4,)
    assert load_labelled_test_dataset().get_data().shape == (4,)
    assert load_labelled_test_dataset().get_label().shape == (4,)
    assert tf.reduce_all(
        tf.equal(
            tf.unique(load_labelled_test_dataset().get_label())[0],
            tf.convert_to_tensor(["maths", "animals"]),
        )
    )
