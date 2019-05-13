#!/usr/bin/env python3.6

import json
import numpy as np
from typing import Iterator, Iterable, List, Dict, Optional
from model_util.preprocessing import Vectorizer
from model_util import util, io_util

logger = util.get_logger('project')


def article_generator(json_file_paths) -> Iterator[str]:
    for json_path in json_file_paths:
        with open(json_path) as f:
            json_obj = json.load(f)
        yield json_obj['article']


class IntegerStream:

    def __init__(self,
                 integers_per_stream: Optional[int],
                 drop_remainder=True):
        if not drop_remainder:
            raise NotImplementedError(
                'drop_remainder=False not implemented yet.')
        self.integers_per_stream = integers_per_stream
        self.drop_remainder = drop_remainder

    def stream(self, data: Iterable[int]) -> Iterator[int]:
        try:
            if self.integers_per_stream is None:
                yield data
                return

            current_stream = []
            for integer in data:
                if len(current_stream) < self.integers_per_stream:
                    current_stream.append(integer)
                if len(current_stream) == self.integers_per_stream:
                    yield current_stream
                    current_stream = []
            if not self.drop_remainder and len(current_stream) > 0:
                logger.trace(f'Yielding remainder: {current_stream}')
                yield current_stream
        except StopIteration:
            return


class VectorizedTextStream:

    def __init__(self,
                 vectorizer: Vectorizer,
                 tokens_per_stream: int = None,
                 lower: bool = True,
                 drop_remainder=True):
        if not drop_remainder:
            raise NotImplementedError(
                'drop_remainder=False not implemented yet.')

        self.vectorizer = vectorizer
        self.lower = lower
        self._int_streamer = IntegerStream(
            integers_per_stream=tokens_per_stream,
            drop_remainder=drop_remainder)

    def stream(self, text: Iterable[str]) -> Iterator[int]:
        for item in text:
            if self.lower:
                item = item.lower()
            ids = self.vectorizer.text_to_vector(item)
            yield from self._int_streamer.stream(ids)


class VectorizedTruncatingSentenceStream(VectorizedTextStream):
    """
    Each stream represents a possibly truncated sentence. For any given
    sentence there is exactly 1 element yielded containing at most max_seq_len
    tokens.
    """

    def stream(self, text: Iterable[str]):
        for doc in text:
            for sentence in self.vectorizer.sent_tokenize(doc):
                yield next(super().stream([sentence]))

    def unvectorized_stream(self, text: Iterable[str]):
        for doc in text:
            for sentence in self.vectorizer.sent_tokenize(doc):
                yield self.vectorizer.word_tokenize(sentence)


class VectorizedJSONArticleStream(VectorizedTextStream):

    def stream(self, json_file_paths) -> Iterator[int]:
        if isinstance(json_file_paths, str):
            json_file_paths = [json_file_paths]
        article_text_gen = article_generator(json_file_paths)
        yield from super().stream(text=article_text_gen)


class VectorizedTruncatingJSONArticleStream(VectorizedTruncatingSentenceStream):

    def stream(self, json_file_paths) -> Iterator[int]:
        if isinstance(json_file_paths, str):
            json_file_paths = [json_file_paths]
        article_text_gen = article_generator(json_file_paths)
        yield from super().stream(text=article_text_gen)


class BatchedArticleStream:

    def __init__(self, vectorizer, batch_size, max_seq_len):
        self.vectorizer = vectorizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self._article_streamer = VectorizedJSONArticleStream(
            self.vectorizer, self.max_seq_len)

    def vectorized_stream(self, json_paths):
        current_streams = [
            self._article_streamer.stream(next(json_paths))
            for _ in range(self.batch_size)]
        while True:
            current_batch = []
            for i in range(self.batch_size):
                try:
                    int_stream = next(current_streams[i])
                except StopIteration:
                    # We've reached the end of tokens for this file.
                    # Sample the next path.
                    current_streams[i] = self._article_streamer.stream(
                        next(json_paths))
                    int_stream = next(current_streams[i])
                current_batch.append(int_stream)
            k = np.asarray(current_batch, dtype=np.int64)
            assert tuple(k.shape) == (self.batch_size, self.max_seq_len), k.shape
            yield k

    def raw_text_stream(self, json_paths):
        for vectors in self.vectorized_stream(json_paths):
            yield self.vectorizer.sequences_to_docs([
                [x] for x in vectors
            ])