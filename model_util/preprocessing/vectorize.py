"""Classes and functions for sentence segmentation, word tokenization, as well
as helpers for converting tokenized sentences to integer id sequences, and other
operations on tokenized text.
"""

from __future__ import print_function

import re
import nltk
import numpy as np
from operator import itemgetter
from functools import partial
from collections import OrderedDict, Counter
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer

from .. import util

logger = util.get_logger('lm.vectorizer')


def _tokenize_docs_fn(docs, self):
    """Parallelized implementation for WordTokenizer.
    See WordTokenizer.tokenize for more info.
    """
    return [self.tokenize(doc) for doc in docs]
tokenize_docs_fn = util.parallelize(_tokenize_docs_fn)


class WordTokenizer:

    def __init__(self):
        # The tokenizer used by nltk.word_tokenize under-the-hood.
        # To emphasize: this is *identical* to using word_tokenize.
        # I just like having it explicitly here for clarity.
        self._tokenizer = nltk.tokenize.TreebankWordTokenizer()
        # (experimental)
        self._detokenizer = nltk.tokenize.treebank.TreebankWordDetokenizer()

    def tokenize(self, s, preserve_line=False):
        """Same layout as nltk.word_tokenize."""
        if preserve_line:
            sentences = [s]
        else:
            sentences = nltk.sent_tokenize(s, language='english')
        return [token for sent in sentences
                for token in self._tokenizer.tokenize(sent)]

    def tokenize_docs(self, docs, parallel=True):
        if parallel:
            return tokenize_docs_fn(docs, self=self)
        else:
            return [self.tokenize(d) for d in docs]

    def detokenize(self, tokens):
        # TODO: test.
        # First call the default detokenizer, then we'll clean up
        # some mistakes I've observed.
        res = self._detokenizer.detokenize(tokens)
        # Remove spaces before punctuation.
        res = re.sub(r'(\w) ([.?!,])', r'\1\2', res)
        # Insert space after commas.
        res = re.sub(r'([,])(\w)', r'\1 \2', res)
        return res


def _count_tokens(tokenized_docs, binary_doc_counts=False):
    """Count tokens occurring in all docs in `tokenized_docs`.

    Args:
        tokenized_docs: list of word-tokenized documents.
        binary_doc_counts: If True, don't count a given token more than once
            for a given doc. (doc_tokens => set(doc_tokens))
    
    Returns:
        list of Counters, each of which maps words to counts for a given doc.
        They will be merged (via adding counts) in fit_on_docs.
    """
    word_counts = []
    for tokens in tokenized_docs:
        # Instantiate counter for this doc.
        tmp = Counter()
        # Determine what to but in our bag for counting tokens.
        # i.e. whether we care about duplicates.
        bag = tokens
        if binary_doc_counts:
            bag = set(bag)
        # Count away!
        for w in bag:
            tmp[w] += 1
        word_counts.append(tmp)
    return word_counts
word_counts_fn = util.parallelize(partial(_count_tokens, binary_doc_counts=False))
doc_freqs_fn = util.parallelize(partial(_count_tokens, binary_doc_counts=True))


class OrderedCounter(OrderedDict, Counter):
    """Counter that remembers the order elements are first encountered.
    Source: https://docs.python.org/3.4/library/collections.html?highlight=ordereddict#ordereddict-examples-and-recipes
    """

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class SimpleVectorizer:
    """Learns a dictionary from a given set of tokenized documents,
    and provides a user-friendly interface for subsequent conversions from
    tokens --> integer sequences.
    """
    # Tokens (and their IDs) always placed at the top of any vocab list.
    # Note: These are counted as part of the vocabulary.
    _PAD_TOKEN = '_PAD'
    _UNK_TOKEN = '_UNK'
    START_VOCAB = OrderedDict([(_PAD_TOKEN, 0), (_UNK_TOKEN, 1)])
    # Tokens with global frequency < FREQ_CUTOFF will be ignored entirely.
    # Specifically, we won't even insert UNK for tokens that occurred less than
    # this number of times across the entire set of calls to update().
    FREQ_CUTOFF = 0

    def __init__(self, vocab_size=None):
        self.vocab_size = vocab_size

        # Tokenization tools and other private attributes.
        self._sent_tokenizer = PunktSentenceTokenizer()
        self._sent_trainer = PunktTrainer()
        self._word_tokenizer = WordTokenizer()
        self._index_to_dfreq = None
        self._is_finalized = False

        # Number of times a given word has been seen across entire corpus.
        self.word_to_freq = OrderedCounter()
        # Number of docs that contained word w.
        self.word_to_dfreq = OrderedCounter()
        # Number of documents trained on so far.
        self.num_docs = 0

        # Dicts that will be filled when fitting documents.
        # word_index: w => i (index into vocabulary)
        # index_docs: i => doc_counts (doc_freq for word with index i).
        self.word_to_index = OrderedDict()

    def truncate_vocab(self, new_vocab_size=None):
        # If not given, set to exact number of unique tokens we've seen.
        new_vocab_size = new_vocab_size or len(self.word_to_index)
        logger.debug(f'Truncating vocab size from {self.vocab_size} '
                     f'to {new_vocab_size}.')

        if new_vocab_size > len(self.word_to_index):
            raise ValueError(
                'truncate_vocab received new vocab larger than previous.')
        self.vocab_size = new_vocab_size
        self.word_to_index = OrderedDict(
            (k, v) for k, v in self.word_to_index.items()
            if v < self.vocab_size)

        assert len(self.word_to_index) == self.vocab_size, '{} != {}'.format(
            len(self.word_to_index), self.vocab_size)

    @property
    def vocab(self):
        vocab = list(self.word_to_index.keys())
        expected_len = self.vocab_size
        if self.vocab_size is not None and len(vocab) != expected_len:
            raise RuntimeError(
                'Vectorizer\'s word_to_index dictionary has unexpected number of entries:'
                ' {}. It should have {}.'.format(len(vocab), expected_len))
        return vocab

    @util.timed_function()
    def update(self, tokenized_docs):
        if self._is_finalized:
            raise RuntimeError('Vectorizer has been finalized. Update prohibited.')

        # Compute frequency statistics.
        self.word_to_freq.update(Vectorizer.get_word_dict(
            'word_to_freq', tokenized_docs))
        self.word_to_dfreq.update(Vectorizer.get_word_dict(
            'word_to_dfreq', tokenized_docs))

        logger.info('Longest sequence in docs has {} tokens.'.format(
            len(max(tokenized_docs, key=len))))
        logger.info(f'Num unique tokens seen thus far:'
                    f' {len(self.word_to_freq)}')
        logger.info(f'Num tokens total seen thus far:'
                    f' {sum(self.word_to_freq.values())}')

        if self.vocab_size is None:
            common_words_sorted = util.lmap(
                itemgetter(0), self.word_to_freq.most_common())
        else:
            common_words_sorted = util.lmap(
                itemgetter(0), self.word_to_freq.most_common(
                    self.vocab_size - len(self.START_VOCAB)))

        assert '_PAD' not in common_words_sorted, [
            l for l in tokenized_docs if '_PAD' in l
        ][:2]
        assert '_UNK' not in common_words_sorted

        word_start = len(self.START_VOCAB)
        word_end = self.vocab_size or (word_start + len(common_words_sorted))
        self.word_to_index = OrderedDict(
            list(self.START_VOCAB.items()) +
            list(zip(common_words_sorted, range(word_start, word_end))))

    def finalize_updates(self):
        """Teardown operations after last call to .update, as determined
        by user.
        """
        self._finalize_frequency_dicts()
        self._finalize_sent_tokenizer()
        self._is_finalized = True

    def _finalize_frequency_dicts(self):
        # Formally insert entries for START_VOCAB in self.word_to_freq,
        # and align items with self.word_to_index.
        unk_freqs = sum([f for w, f in self.word_to_freq.items()
                         if w not in self.word_to_index])
        self.word_to_freq = OrderedCounter(
            [(self._PAD_TOKEN, None), (self._UNK_TOKEN, unk_freqs)] +
            [(w, self.word_to_freq[w]) for w in self.word_to_index])
        # Not obvious how to do analogous procedure for doc freqs, so set both
        # special token counts to None.
        self.word_to_dfreq = OrderedCounter(
            [(self._PAD_TOKEN, None), (self._UNK_TOKEN, None)] +
            [(w, self.word_to_dfreq[w]) for w in self.word_to_index])

    def _finalize_sent_tokenizer(self):
        """Re-instantiate sentence tokenizer to ensure has updated params."""
        self._sent_tokenizer = PunktSentenceTokenizer(
            self._sent_trainer.get_params())

    def tokens_to_vector(self, tokens):
        """Converts list of word tokens to list of integer ids."""
        sent_vec = []
        for word in tokens:
            if self.is_common_unknown(word):
                sent_vec.append(self.START_VOCAB['_UNK'])
            elif self.is_in_vocabulary(word):
                sent_vec.append(self.word_to_index.get(word))
        return sent_vec

    def is_in_vocabulary(self, word):
        return word in self.word_to_index

    def is_common_unknown(self, word):
        return not self.is_in_vocabulary(word) and \
               self.word_to_freq.get(word, 0) >= self.FREQ_CUTOFF

    def detokenize(self, tokens):
        return self._word_tokenizer.detokenize(tokens)

    @util.listify
    def sent_tokenize(self, docs):
        """
        Args:
            docs: str or list(str)

        Returns:
            docs, with each entry tokenized into sentence strings.
        """
        return self._sent_tokenizer.tokenize_sents(docs)

    @util.listify
    def word_tokenize(self, docs, parallel=True):
        return self._word_tokenizer.tokenize_docs(docs, parallel=parallel)


class Vectorizer(SimpleVectorizer):
    """Responsible for transforming texts (docs) into numerical sequences,
    vectors, matrices, etc.

    Extends SimpleVectorizer to support updating on non-tokenized docs,
    and for converting documents into matrices (sentence by token).
    """
    FREQ_CUTOFF = 0

    def __init__(self, max_num_sents, max_seq_len, vocab_size):
        """
        Args:
            vocab_size: total number of (unique) words to use.
        """
        super(Vectorizer, self).__init__(vocab_size)
        self.max_num_sents = max_num_sents
        self.max_seq_len = max_seq_len

    @util.timed_function()
    def update(self, docs):
        """Updates internal vocabulary based on a list of texts.

        Args:
            docs: list of text strings.
        
        Returns:
            stats: dictionary of common statistics. Has keys:
                ['word_to_index', 'index_to_word', 'word_to_freq',
                'word_to_dfreq', 'index_to_dfreq']
        """
        docs = list(docs)

        if self._is_finalized:
            raise RuntimeError('Vectorizer has been finalized. Update prohibited.')

        # Tokenize docs.
        self.num_docs += len(docs)
        logger.debug('fit_on_docs beginning with {} docs.'.format(len(docs)))
        self._train_sent_tokenizer(docs)
        docs_tokenized = self.word_tokenize(docs)

        # Feed tokenized docs to SimpleVectorizer.update.
        super(Vectorizer, self).update(docs_tokenized)

        # Ensure postconditions are satisfied.
        assert len(self.word_to_index) <= self.vocab_size  # Sanity check.
        logger.debug('fit_on_docs ending with {} docs.'.format(
            len(docs_tokenized)))

    # @util.timed_function()
    def docs_to_sequences(self, docs):
        return list(self.docs_to_sequences_generator(docs))

    def docs_to_sequences_generator(self, docs):
        self._finalize_sent_tokenizer()
        for doc in self.sent_tokenize(docs):
            # Possible fix: Move this below the _single_doc_to_sequences call.
            if len(doc) == 0:
                logger.debug('Skipping empty doc.')
                continue
            yield self._single_doc_to_sequences(doc)

    def _single_doc_to_sequences(self, doc):
        """list(sent-str) => list(list(int))."""
        sentence_vectors = []
        for tokens in self.word_tokenize(doc, parallel=False):
            sent_vec = self.tokens_to_vector(tokens)
            if len(sent_vec) > 0:
                sentence_vectors.append(sent_vec)
            if len(sentence_vectors) == self.max_num_sents:
                break
        return sentence_vectors

    def tokens_to_vector(self, tokens):
        """Converts list of word tokens to list of integer ids."""
        if len(self.word_to_index) != self.vocab_size:
            raise ValueError(
                'Cannot vectorize tokens before training. '
                'You may also receive this error if your vocab size '
                'exceeds the total number of unique tokens in your corpus.')

        vec = super(Vectorizer, self).tokens_to_vector(tokens)
        return vec[:self.max_seq_len]

    def text_to_vector(self, text):
        self._finalize_sent_tokenizer()
        sentences = list(filter(lambda s: len(s) > 0, self.sent_tokenize(text)))
        return [tok for sent in self._single_doc_to_sequences(sentences) for tok in sent]

    def sentence_to_vector(self, sentence):
        tokens = self.word_tokenize(sentence, parallel=False)
        return self.tokens_to_vector(tokens)

    def sequences_to_docs(self, sequences):
        """Detokenization of integer sequences.
        The inverse operation of docs_to_sequences.
        """
        return [self._sequence_to_doc(seq) for seq in sequences]

    def _sequence_to_doc(self, sequence):
        """
        Args:
            sequence: (~N, ~T)
        """
        doc = []
        for int_tokens in sequence:
            word_tokens = [self.index_to_word.get(i) for i in int_tokens]
            words_detokenized = self._word_tokenizer.detokenize(word_tokens)
            doc.append(words_detokenized)
        return ' '.join(doc)

    def _train_sent_tokenizer(self, docs):
        self._sent_trainer.train('\n'.join(docs), finalize=False)

    @property
    def index_to_word(self):
        if util.is_empty(self.word_to_index, none_ok=True):
            return None
        return util.swap_key_vals(self.word_to_index)

    @property
    def index_to_dfreq(self):
        if any((util.is_empty(self.word_to_dfreq, none_ok=True),
                self.index_to_word is None)):
            return None
        elif self._index_to_dfreq is not None:
            return self._index_to_dfreq

        self._index_to_dfreq = OrderedCounter()
        for word, dfreq in list(self.word_to_dfreq.items()):
            index = self.word_to_index.get(word)
            if index is not None:
                self._index_to_dfreq[index] = dfreq
        return self._index_to_dfreq

    @staticmethod
    def get_word_dict(name, word_tokenized_docs):
        """
        Args:
            word_tokenized_docs:
            name: any of the keys in name_to_fn (next line)
        """
        name_to_fn = {'word_to_freq': 'word_counts_fn',
                      'word_to_dfreq': 'doc_freqs_fn'}

        dictionaries = globals()[name_to_fn[name]](word_tokenized_docs)
        # Combine the list of dictionaries.
        # Note that they're Counters, which has `update` defined as
        # adding counts (slightly different than dict.update).
        _word_dict = Counter()
        for d in dictionaries:
            _word_dict.update(d)
        return _word_dict


class PaddingVectorizer(Vectorizer):
    """Vectorizer capable of dynamically padding vectorized sentences.

    NOTE: haven't decided how to deal with users calling docs_to_matrix multiple
    times, in terms of padding reasons. The concern is that it would probably
    return matrices with different dynamically-padded shapes from each call, which
    could be problematic depending on the user's use case.
    """
    # Nonconfigurable. Absolute maximums that will never be crossed.
    # These protect users from themselves.
    _MAXIMUM_MAX_SEQ_LEN = 150
    _MAXIMUM_MAX_NUM_SENTS = 500

    def __init__(self, max_num_sents, max_seq_len, vocab_size):
        super(PaddingVectorizer, self).__init__(
            max_num_sents, max_seq_len, vocab_size)
        self._max_seq_lens = []
        self._max_num_sents = []

    @util.timed_function()
    def update(self, docs):
        super(PaddingVectorizer, self).update(docs)
        _docs = [self.word_tokenize(doc, parallel=False) for doc in self.sent_tokenize(docs)]
        if self.max_seq_len is None:
            self._max_seq_lens.append(PaddingVectorizer.true_max_seq_len(_docs))
        if self.max_num_sents is None:
            self._max_num_sents.append(PaddingVectorizer.true_max_num_sents(_docs))

    def finalize_updates(self):
        super(PaddingVectorizer, self).finalize_updates()
        if self.max_seq_len is None:
            self.max_seq_len = min(int(np.median(self._max_seq_lens)),
                                   self._MAXIMUM_MAX_SEQ_LEN)
            logger.debug('Assigned max_seq_len={}'.format(self.max_seq_len))
        if self.max_num_sents is None:
            self.max_num_sents = min(int(np.median(self._max_num_sents)),
                                     self._MAXIMUM_MAX_NUM_SENTS)
            logger.debug('Assigned max_num_sents={}'.format(self.max_num_sents))

    def docs_to_matrix(self, docs, _print_info=None):
        # Convert each doc to list of sentence vectors.
        docs_sequences = self.docs_to_sequences(docs, _print_info=_print_info)
        padded_sequences = self.pad(docs_sequences)
        return np.stack(padded_sequences)

    def tokens_to_vector(self, tokens):
        unpadded_vec = super(PaddingVectorizer, self).tokens_to_vector(tokens)
        return self.pad_vector(unpadded_vec)

    def pad(self, arr, max_seq_len=None, max_num_sents=None):
        num_dims = PaddingVectorizer._get_num_dims(arr)

        if num_dims == 1:
            return self.pad_vector(arr, max_seq_len)
        elif num_dims == 2:
            return self.pad_matrix(arr, max_num_sents, max_seq_len)

        padded_sub_arrays = []
        for sub_array in arr:
            padded_sub_arrays.append(self.pad(
                sub_array, max_seq_len, max_num_sents))
        return padded_sub_arrays

    def pad_vector(self, vector, max_seq_len=None):
        max_seq_len = max_seq_len or self.max_seq_len
        vector = np.asarray(vector[:max_seq_len])
        if len(vector) < max_seq_len:
            padding = [self.START_VOCAB['_PAD']] * (max_seq_len - len(vector))
            vector = np.concatenate((vector, padding))

        if vector.shape != (max_seq_len,):
            raise RuntimeError('Padded vector resulted in shape {}, but expected {}.'.format(
                vector.shape, (max_seq_len,)))

        return vector

    def pad_matrix(self, matrix, max_num_sents=None, max_seq_len=None):
        max_num_sents = max_num_sents or self.max_num_sents
        max_seq_len = max_seq_len or self.max_seq_len

        padded_matrix = []
        for row in matrix[:max_num_sents]:
            padded_matrix.append(self.pad_vector(row, max_seq_len))

        if len(padded_matrix) < max_num_sents:
            padded_matrix = self._append_padding_sents(
                padded_matrix,
                max_seq_len=max_seq_len,
                max_num_sents=max_num_sents)

        padded_matrix = np.asarray(padded_matrix)
        if padded_matrix.shape != (max_num_sents, max_seq_len):
            raise RuntimeError('Padded matrix resulted in shape {}, but expected {}.'.format(
                padded_matrix.shape, (max_num_sents, max_seq_len)))
        return padded_matrix

    def _append_padding_sents(self, matrix, max_seq_len, max_num_sents):
        num_padding_sents = max_num_sents - len(matrix)
        padding_matrix = self._get_padding(shape=(num_padding_sents, max_seq_len))
        return np.concatenate([matrix, padding_matrix])

    def _get_padding(self, shape):
        pad_value = self.START_VOCAB['_PAD']
        padding = np.zeros(shape=shape, dtype=type(pad_value))
        padding.fill(pad_value)
        return padding

    @staticmethod
    def _get_num_dims(arr):
        num_dims = 0
        current_arr = arr
        while isinstance(current_arr, util.LIST_TYPES):
            num_dims += 1
            if current_arr == []:
                break
            current_arr = current_arr[0]
        return num_dims

    @classmethod
    def true_max_num_sents(cls, arr):
        num_dims = PaddingVectorizer._get_num_dims(arr)
        if num_dims == 1:
            return 1
        elif num_dims == 2:
            return len(arr)
        return max([PaddingVectorizer.true_max_num_sents(sub_arr) for sub_arr in arr])

    @classmethod
    def true_max_seq_len(cls, arr):
        num_dims = PaddingVectorizer._get_num_dims(arr)
        if num_dims == 1:
            if Vectorizer.START_VOCAB['_PAD'] in arr:
                return list(arr).index(Vectorizer.START_VOCAB['_PAD'])
            return len(arr)
        return max([PaddingVectorizer.true_max_seq_len(sub_arr) for sub_arr in arr])

