#!/usr/bin/env python3.6

import os
import shutil
import argparse
import pickle
import numpy as np
import tensorflow as tf

import turibolt as bolt

from tensorflow.python import keras
from model_util import io_util, util
from operator import itemgetter

from model_util.preprocessing import ALLOWED_KINDS
from model_util.preprocessing import path_generator as pg
from model_util.preprocessing import Vectorizer

from stream import BatchedArticleStream, article_generator
from callbacks import LoggingCallback
from trainer import Trainer

# tf.compat.v1.logging.set_verbosity('INFO')
assert tf.executing_eagerly()

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_level', default=None, type=str,
    help='Global logging level. "info", "debug", etc.')
parser.add_argument(
    '--bundle', default=None, type=str)
parser.add_argument(
    '-c', '--config', default='test',
    help='Basename of config file in config directory. '
         'Example: example_config')
parser.add_argument(
    '-d', '--data_dir', default='/Users/brandon/Downloads/foxnews.com')
parser.add_argument(
    '--warm_start', action='store_true', default=False)
parser.add_argument(
    '--load_vectorizer', action='store_true', default=False)
args = parser.parse_args()

# Build up path to custom config file.
root_dir = io_util.Directory.of_file(__file__)
if os.path.exists(args.config):
    config_path = args.config
else:
    config_path = root_dir.subdir('config').join(args.config)
    config_path = util.force_extension(config_path, 'yml')

# Merge custom config file with command-line arguments.
config = util.parse_config(args=args, config_path=config_path)

logger = util.get_logger(
    'project',
    is_root_logger=True,
    console_level=config.log_level)

if args.bundle is None:
    args.bundle = root_dir.subdir('models', create=True).join('default')

bundle = io_util.Directory(args.bundle)
if not bundle.exists():
    logger.info(f'Creating bundle: {bundle}')
    bundle.create()
elif not any((args.warm_start, args.load_vectorizer)):
    logger.info(f'Clearing previous bundle: {bundle}')
    bundle.delete_contents()

if util.is_running_on_bolt():
    bundle = io_util.Directory(bolt.ARTIFACT_DIR).subdir(
        'bundle', create=True)


def get_vectorizer(path_generator=None) -> Vectorizer:

    def get_kind(kind):
        return [a.lower() for a in article_generator(
            path_generator.get_paths(kind))]

    if args.warm_start or args.load_vectorizer:
        logger.warning('Loading vectorizer.')
        with open(bundle.get_file_path('vectorizer.pkl'), 'rb') as f:
            return pickle.load(f)
    else:
        # Initialize vectorizer.
        vectorizer = Vectorizer(
            max_num_sents=None,
            max_seq_len=None,
            vocab_size=config.vocab_size)
        # vectorizer.FREQ_CUTOFF = 1
        logger.trace('Training vectorizer.')
        vectorizer.update(get_kind('train')[:config.find('max_samples')])
        vectorizer.finalize_updates()
        vectorizer.truncate_vocab()
        with open(bundle.join('vocab.txt'), 'w+') as f:
            for word in vectorizer.vocab:
                f.write(word + '\n')
        with open(bundle.join('vectorizer.pkl'), 'wb+') as f:
            pickle.dump(vectorizer, f)
        return vectorizer


def replace_with_unk(int_sequence, vocab_size, unk_id):
    # Replace any OOV tokens with UNK token ID.
    unk_repeated = tf.tile(
        tf.convert_to_tensor(value=[unk_id], dtype=tf.int64),
        tf.shape(input=int_sequence, out_type=tf.int64))
    int_sequence = tf.where(
        int_sequence < vocab_size,
        int_sequence, unk_repeated)
    return int_sequence


def count(ds):
    n = 0
    for _ in ds:
        n += 1
    return n


num_tokens_total = util.DotDict({
    'cnn': {
        'train': 3667696,
        'valid': 211223,
    }
})


def old_get_ds(kind, vectorizer, path_generator):
    from model_util.preprocessing import tf_util
    # Elements are text lines, each representing a full article.
    ds = tf.data.TextLineDataset(list(path_generator.get_paths(kind)))
    ds = ds.map(lambda x: tf_util.parse_text_line(
        x,
        vocab_size=vectorizer.vocab_size,
        max_seq_len=config.max_seq_len + 1,
        unk_id=vectorizer.START_VOCAB['_UNK']))
    ds = ds.batch(config.find('batch_size'))
    return ds


def batch_to_text(batch, vectorizer):
    return vectorizer.sequences_to_docs([
        [elem] for elem in batch])


def get_ds(kind, vectorizer, path_generator):
    """TODO: test this new impl."""
    # Elements are text lines, each representing a full article.
    ds = tf.data.TextLineDataset(list(path_generator.get_paths(kind)))

    # Shuffle & tokenize the articles.
    # Now each element is list of integer IDs for a full article.
    ds = ds.shuffle(buffer_size=4)\
        .map(lambda line: tf.strings.split([line]).values)\
        .map(lambda tokens: tf.strings.to_number(tokens, tf.int64))\
        .map(lambda ids: replace_with_unk(
            ids, config.vocab_size,
            unk_id=vectorizer.START_VOCAB['_UNK']))

    # Flatten. Now each element is a single integer ID.
    ds = ds.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))

    if kind not in num_tokens_total.cnn:
        num_tokens_total.cnn[kind] = count(ds)
    logger.info(f'Num {kind} tokens: {num_tokens_total.cnn[kind]}')

    # Split into batch_size total windows.
    batch_size = config.find('batch_size')
    T = num_tokens_total.cnn[kind] // batch_size
    logger.info(f'Tokens per batch elem per corpus: {T}')
    ds = ds.batch(T, drop_remainder=True)
    ds = ds.batch(batch_size)

    # Entire dataset is now a singe element with shape (B, T).
    for monolithic_single_tensor in ds:
        t = 0
        while t + config.max_seq_len < T:
            yield monolithic_single_tensor[:, t:t+config.max_seq_len], \
                  monolithic_single_tensor[:, t+1:t+1+config.max_seq_len]

            t += config.max_seq_len
            if config.search('max_steps') \
                    and t // config.max_seq_len >= config.find('max_steps'):
                break


def get_model() -> keras.Sequential:
    if args.warm_start:
        logger.warning('Warm starting the model.')
        return keras.models.load_model(bundle.get_file_path('model.h5'))
    else:
        layers = [
            keras.layers.Embedding(
                **config.graph.embedding,
                batch_input_shape=(config.find('batch_size'), None))]
        for _ in range(config.graph.num_lstm_layers):
            layers.append(keras.layers.LSTM(**config.graph.lstm))
        layers.append(keras.layers.Dense(**config.graph.dense))
        return keras.Sequential(layers)


class GenerateTextCallback(keras.callbacks.Callback):

    def __init__(self, vectorizer: Vectorizer, num_tokens=20, seeds=None):
        super().__init__()
        self.vectorizer = vectorizer
        self.num_tokens = num_tokens
        self.seeds = seeds or [
            (vectorizer.vocab[i],) for i in range(7)
        ]

    def on_epoch_end(self, epoch, logs=None):
        for seed in self.seeds:
            self.model.reset_states()

            # Initialize by making prediction on seed token(s).
            current_seq = [self.vectorizer.word_to_index[tok] for tok in seed]
            inputs = self._get_batch(current_seq)
            probs = self.model.predict_on_batch(inputs)[0]
            current_seq.append(probs[-1].argmax())

            for t in range(self.num_tokens):
                inputs = self._get_batch([current_seq[-1]])
                probs = self.model.predict_on_batch(inputs)[0]
                probs = probs.squeeze()
                current_seq.append(probs.argmax())

            text_rant = self.vectorizer.sequences_to_docs([[current_seq]])
            logger.info(text_rant)

    def _get_batch(self, arr):
        batch = np.zeros((config.find('batch_size'), len(arr)))
        batch[0, :] = arr
        return batch


def main():
    util.print_box('Parameters', sorted(config.items(), key=itemgetter(0)))

    data_dir = io_util.Directory(args.data_dir)
    args.data_dir = io_util.Directory(args.data_dir)
    path_generator = pg.FileFlatGenerator(data_dir)
    util.print_box('Files per kind', [
        (k, path_generator.num_files(k))
        for k in ALLOWED_KINDS])
    vectorizer = get_vectorizer(path_generator)
    config.update_recursively(vocab_size=vectorizer.vocab_size)
    config.graph.dense.units = config.vocab_size
    config.graph.embedding.input_dim = config.vocab_size
    util.save_yaml_config(config, bundle.join('config.yml'))

    tr = Trainer(get_model(), model_dir=bundle, params=config.trainer)
    tr.model.summary()
    util.print_box('Weights', [
        (w.name, w.shape) for w in tr.model.weights
    ])

    exit(0)
    try:
        tr.train_and_evaluate(
            train_input_fn=lambda: get_ds('train', vectorizer, path_generator),
            valid_input_fn=lambda: get_ds('valid', vectorizer, path_generator))
    except KeyboardInterrupt:
        logger.warning('Training terminated.')

    del tr

    model = keras.models.load_model(bundle.get_file_path('model.h5'))
    cb = GenerateTextCallback(vectorizer, seeds=[('beyer', "'s")])
    cb.set_model(model)
    cb.on_epoch_end(None, None)

    if util.is_running_on_bolt():
        shutil.copytree(
            src=str(bundle),
            dst=bundle.join('final'))


if __name__ == '__main__':
    main()

