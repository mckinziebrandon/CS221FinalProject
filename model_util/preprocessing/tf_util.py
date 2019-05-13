import tensorflow as tf
import numpy as np
import sys

from .. import util

INT_TYPES = (int, np.int, np.int16, np.int32, np.int64)
FLOAT_TYPES = (float, np.float, np.float32, np.float64, np.float128)

logger = util.get_logger('project.model_util.preprocessing.tf_util')


def to_feature(values):
    """Wrap values inside tf.train.Feature of appropriate type.

    Args:
        values: list(int/float/str).

    Returns:
        tf.train.Feature
    """
    if isinstance(values[0], INT_TYPES):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
    elif isinstance(values[0], FLOAT_TYPES):
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))
    elif isinstance(values[0], str):
        values = [bytes(item, 'utf-8') for item in values]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))
    elif isinstance(values[0], bytes):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))
    else:
        raise ValueError(
            "Values not a recognized type; v: %s type: %s" %
            (str(values[0]), str(type(values[0]))))


def to_feature_list(values):
    """
    Args:
        values: list(list(int/float/str))
    """
    return tf.train.FeatureList(feature=[
        to_feature(v) for v in values])


def to_features(dictionary):
    """Helper: build tf.train.Features from str => list(int/float/str) dict."""
    features = {}
    for k, v in dictionary.items():
        if not v:
            raise ValueError("Empty generated field: %s", str((k, v)))
        features[k] = to_feature(v)
    return tf.train.Features(feature=features)


def to_feature_lists(dictionary):
    """
    Args:
        dictionary: str => list(list(int/float/str))
    """
    feature_list = {}
    for k, values in dictionary.items():
        feature_list[k] = to_feature_list(values)
    return tf.train.FeatureLists(feature_list=feature_list)


def parse_text_line(line, vocab_size, max_seq_len=-1, unk_id=3):
    """Used if tf_dataset == TextLineDataset.

    This processes a single line, assumed to be a sequence of
    integer IDs, into a 1D int64 `Tensor`.

    Args:
        line: tf.string `Tensor`.
        vocab_size:
        max_seq_len:
        unk_id:

    Returns:
        (features, labels) where both are int64 `Tensors` of same size.
    """
    string_elems = tf.strings.split([
        tf.strings.strip(line + ' 0'),
    ], maxsplit=max_seq_len or -1).values[:-1]
    int_sequence = tf.strings.to_number(string_elems, tf.int64)

    # Replace any OOV tokens with UNK token ID.
    unk_repeated = tf.tile(
        tf.convert_to_tensor(value=[unk_id], dtype=tf.int64),
        tf.shape(input=int_sequence, out_type=tf.int64))
    int_sequence = tf.where(
        int_sequence < vocab_size,
        int_sequence, unk_repeated)

    logger.trace(f'INT SEQ: {int_sequence}')
    return int_sequence[:-1], int_sequence[1:]


def tf_print(name, tensor, summarize=None):
    """More intuitive and easier to use wrapper around tf.print, provided:
        - You just want to print the value of a tensor and a name.

    Args:
        name: (str) name to print to the left of the tensor on stdout. Note that
            this will also be the name of the returned tensor, which just
            wraps the input `tensor` with tf.identity.
        tensor: (tf.Tensor) that we want printed and passed through.
        summarize: The first and last summarize elements within each dimension
            are recursively printed per Tensor. If None, then the first 3 and
            last 3 elements of each dimension are printed for each tensor.
            If set to -1, it will print all elements of every tensor.

    Returns:
        tf.identity(`tensor`) of the input `tensor`. Required for the
        print op to be executed. You must use the returned tensor in place
        of the input `tensor` in your subsequent code!
    """
    print_op = tf.print(name, tensor, output_stream=sys.stdout, summarize=summarize)
    with tf.control_dependencies([print_op]):
        tensor = tf.identity(tensor, name)
    return tensor


def parse_tfrecord(raw_example, features_name, labels_name):
    """Used if tf_dataset == TFRecordDataset.

    This processes a serialized TFRecord produced by LMDataset.

    Args:
        raw_example:
        features_name: (str)
        labels_name: (str)

    Returns:
        (features, labels) where both are int64 `Tensors` of same size.
    """
    feature_spec = {
        name: tf.io.FixedLenSequenceFeature((), tf.int64, True)
        for name in [features_name, labels_name]}
    parsed_example = tf.io.parse_single_example(
        serialized=raw_example,
        features=feature_spec)
    labels = parsed_example.pop(labels_name)
    features = parsed_example.pop(features_name)
    with tf.control_dependencies([
        tf.compat.v1.assert_equal(tf.shape(input=features), tf.shape(input=labels))
    ]):
        return features, labels
