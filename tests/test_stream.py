
import tensorflow as tf
from . import testutils

import h5py
import os
import unittest
import numpy as np
from operator import itemgetter

import stream
from stream import *
from model_util.preprocessing import *
from model_util.preprocessing.path_generator import *

logger = util.get_logger(
    'project',
    is_root_logger=True,
    console_level='info')


class TestStream(unittest.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.max_seq_len = 3
        self.num_epochs = 2
        self.vocab_size = 100
        self.data_dir = testutils.get_resource_dir('fake_data')

        self.vectorizer = Vectorizer(
            max_num_sents=None,
            max_seq_len=None,
            vocab_size=self.vocab_size)

        self.pg = FileFlatGenerator(self.data_dir, data_split=util.DotDict({
            'train': 1.0, 'valid': 0.0, 'test': 0.0
        }))

        train_paths = list(self.pg.get_paths('train'))
        assert len(train_paths) == 4, train_paths
        self.vectorizer.update([
            a.lower() for a in stream.article_generator(iter(train_paths))])
        self.vectorizer.finalize_updates()
        self.vectorizer.truncate_vocab()

    def _get_fake_data(self, filename) -> str:
        path = testutils.get_resource_dir('fake_data').get_file_path(filename)
        text = next(stream.article_generator([path]))
        return text

    def test_article_stream(self):
        streamer = stream.BatchedArticleStream(
            self.vectorizer,
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len)
        batch_stream = streamer.raw_text_stream(self.pg.get_paths(
            'train', max_num_paths=None))

        for i, x in enumerate(batch_stream):
            util.print_box(f'Batch {i}', x)

    def test_vectorized_truncating_sentence_stream(self):
        streamer = stream.VectorizedTruncatingSentenceStream(
            tokens_per_stream=None,
            vectorizer=self.vectorizer)

        articles = [self._get_fake_data(path)
                    for path in testutils.get_resource_dir('fake_data').ls()]

        logger.info('Sentence stream over all articles:')
        for item in streamer.unvectorized_stream(articles):
            print(item)

        logger.info('Vectorized sentence stream over all articles:')
        for item in streamer.stream(articles):
            print(item)

    def test_vectorized_truncating_json_stream(self):
        streamer = stream.VectorizedTruncatingJSONArticleStream(
            vectorizer=self.vectorizer)

        json_file_paths = testutils.get_resource_dir('fake_data').ls()
        logger.info('Vectorized sentence stream over all articles:')
        for item in streamer.stream(json_file_paths):
            print(item)
