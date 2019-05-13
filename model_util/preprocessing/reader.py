import pickle
import numpy as np
from abc import abstractmethod, ABC

from .. import util
from . import PaddingVectorizer

logger = util.get_logger('project.model_util.reader')

ALLOWED_KINDS = ['train', 'valid', 'test']


def check_kind(kind):
    assert kind in ALLOWED_KINDS


def num_lines(file_path):
    """Optimized way of counting lines in large files.

    Copied from: https://stackoverflow.com/questions/9629179
    """
    def blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b: break
            yield b

    with open(file_path, encoding='utf-8', errors='ignore') as f:
        return sum(bl.count('\n') for bl in blocks(f))


class PathSharder:

    def __init__(self, num_shards, path_generator):
        self.num_shards = num_shards
        self.paths = util.DotDict({
            k: list(path_generator.get_paths(k))
            for k in ALLOWED_KINDS})

    def num_files_per_shard(self, kind):
        check_kind(kind)
        return len(self.paths[kind]) // self.num_shards

    def get_paths(self, kind, shard_index):
        check_kind(kind)

        num_files = self.num_files_per_shard(kind)
        begin = shard_index * num_files
        end = begin + num_files
        assert end <= len(self.paths[kind]), f'{end} !<= {len(self.paths[kind])} for kind={kind}'

        return self.paths[kind][begin:end]


class DataReader(ABC):
    """Generates lines of data. Abstracts away/handles how different datasets
    might be laid out."""

    def __init__(self, path_generator):
        self.path_generator = path_generator

    @property
    def input_dir(self):
        return self.path_generator.input_dir

    @abstractmethod
    def get_ids(self, *args, **kwargs):
        """Get lines of data formatted as integer ID sequences.
        Args:
            kind: 'train', 'valid', or 'test'.
            num: number of lines to read.

        Returns:
            list(list(int)) with length equal to `num`.
        """
        pass

    def get_lines(self,
                  kind,
                  max_num_lines=None,
                  skip_empty=True,
                  strip_whitespace=True):
        """
        Args:
            kind: one of ALLOWED_KINDS.
            max_num_lines: maximum number of lines to yield.
            skip_empty: if True, will not yield empty lines.
            strip_whitespace: if True, strip lines before yielding them.

        Yields:
            individual lines.
        """
        check_kind(kind)
        num_lines_yielded = 0
        for path in self.path_generator.get_paths(kind):
            with open(path, encoding='utf-8') as f:
                for line in f:
                    if max_num_lines is not None and \
                            num_lines_yielded >= max_num_lines:
                        break

                    if strip_whitespace:
                        line = line.strip()

                    if skip_empty and len(line) == 0:
                        logger.debug_n(1, 'Skipping empty line.')
                        continue

                    yield line
                    num_lines_yielded += 1

    def num_lines(self, kind):
        """Returns: total number of lines in files corresponding to `kind`."""
        return sum(num_lines(path) for path in self.path_generator.get_paths(kind))


class IDReader(DataReader):
    """Data layout: train, valid, test files.

    Assumes:
    - data is already vectorized into IDs.
    """

    def get_ids(self, kind, max_samples=None):
        ids = util.lmap(
            lambda l: np.fromstring(l.strip(), sep=' ', dtype=int),
            self.get_lines(kind, max_num_lines=max_samples))

        if len(ids) < max_samples:
            logger.warning(f'get_ids({kind}, {max_samples}) only found '
                           f'{len(ids)} samples to use.')
        else:
            assert len(ids) == max_samples, len(ids)
        return ids


class PaddingIDReader(IDReader):
    def get_ids(self, kind, max_samples=None, max_len=10):
        ids = super().get_ids(kind, max_samples=max_samples)
        return util.pad(ids, pad_val=0, max_len=max_len)


class PaddingVectorizerReader(DataReader):

    def __init__(self, path_generator):
        super().__init__(path_generator)
        # Instantiate via load_vectorizer or create_vectorizer.
        self.vectorizer = None

    def get_ids(self, kind, max_samples=None):
        ids = []
        for line in self.get_lines(kind, max_num_lines=max_samples):
            tokens = line.rstrip().split()
            assert len(tokens) > 0, tokens
            # Use PaddingVectorizer to ensure vector of length self.max_seq_len.
            ids.append(self.vectorizer.tokens_to_vector(tokens))
        return np.asarray(ids, dtype=int)

    def create_vectorizer(self, max_seq_len, vocab_size, max_train_samples=1e4):
        vectorizer = PaddingVectorizer(
            max_num_sents=1,  # TODO: is this the correct value to set?
            max_seq_len=max_seq_len,
            vocab_size=vocab_size)
        vectorizer.update(docs=list(self.get_lines('train', max_train_samples)))
        vectorizer.finalize_updates()
        vectorizer.truncate_vocab()
        self.vectorizer = vectorizer

    def load_vectorizer(self, vectorizer_path):
        # Disallow vectorizer to ignore any tokens.
        PaddingVectorizer.FREQ_CUTOFF = 0
        # TODO: use self.pickler here.
        logger.debug('Loading dataset vectorizer from disk.')
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)





class TokenVocabulary(ABC):
    """Simple abstract interface for a vocabulary lookup table."""

    @abstractmethod
    def token_to_id(self, token: str):
        pass

    @abstractmethod
    def id_to_token(self, id: int):
        pass


class Vocabulary(TokenVocabulary):

    def __init__(self, vocab_file, unk_token='_UNK'):
        self.vocab_file = vocab_file
        self.unk_token = unk_token
        self.word_to_id = util.BidirectionalDict()
        with open(self.vocab_file) as f:
            for i, line in enumerate(f):
                parsed_line = line.strip().split()
                if len(parsed_line) == 2:
                    word, id = parsed_line
                    self.word_to_id[word] = int(id)
                else:
                    assert len(parsed_line) == 1
                    word = parsed_line[0]
                    self.word_to_id[word] = i

    def token_to_id(self, token):
        return self.word_to_id.get_forward(token)

    def id_to_token(self, id):
        return self.word_to_id.get_reverse(id, default=self.unk_token)

    def ids_to_words(self, ids):
        return util.lmap(self.id_to_token, ids)

    def batch_ids_to_words(self, ids_batch):
        return util.lmap(self.ids_to_words, ids_batch)

    def to_id(self, word: str) -> int:
        # Reminder: 3 == UNK id.
        return self.word_to_id.get(word, 3)

    def words_to_ids(self, words):
        return util.lmap(self.to_id, words)

    def batch_words_to_ids(self, words_batch):
        return util.lmap(self.words_to_ids, words_batch)
