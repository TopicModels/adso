"""Test load_txt function from data module."""

import tensorflow as tf
from adso.data.Test import (
    load_labelled_test_dataset,
    load_test_dataset,
    _write_test_dataset,
)

_write_test_dataset(force=True)

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
