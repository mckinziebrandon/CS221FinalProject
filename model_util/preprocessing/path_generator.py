import random
from abc import abstractmethod, ABC
from typing import Union

from .. import io_util, util
from . import ALLOWED_KINDS, check_kind, TrainValidTestCounter

logger = util.get_logger('project.model_util.reader')


class PathGenerator(ABC):
    """Interface for providing path_generator to data files. Data can be organized
    on disk in a variety of ways, and this interface unifies into an API
    for clients that want to be able to say "just give me the training files",
    etc.
    """

    @abstractmethod
    def get_paths(self, kind):
        """
        Args:
            kind: one of the elements in ALLOWED_KINDS (train, valid, or test).

        Yields:
            string paths one-by-one for the given `kind`.
        """
        pass


class TrivialPathGenerator(PathGenerator):

    def __init__(self, path):
        self.path = path

    def get_paths(self, kind):
        yield self.path


class LocalPathGenerator(PathGenerator):
    """Paths generated from pre-existing local directory (as opposed to
    fetching/reading from a remote file system like blobby).

    Assumptions:
        1) the data is all contained within some base `input_dir`.
        2) there are 1 or more files available for each of train, valid,
            and test.

    Attributes:
        input_dir: io_util.Directory
    """

    def __init__(self, input_dir):
        self.input_dir = io_util.Directory(input_dir)
        if not self.input_dir.exists():
            raise NotADirectoryError(f'Directory does not exist: {input_dir}')
        if self.input_dir.empty():
            raise RuntimeError(f'input_dir is empty: {input_dir}')

    @abstractmethod
    def get_paths(self, kind):
        pass


class ProcessedDataDirectoryGenerator(LocalPathGenerator):
    """Generator for the ProcessedDataDirectory convention.
    See project.model_util.dataset_base for more details.
    """

    def get_paths(self, kind):
        check_kind(kind)
        yield self.input_dir.get_file_path(
            f'{kind}.tfrecords', require_exists=True)


class FileSplitGenerator(LocalPathGenerator):
    """Files are specified for a train/valid/test split.

    Assumptions:
        1) Single file for each of ALLOWED_KINDS.
        2) File for ALLOWED_KINDS[i] contains ALLOWED_KINDS[i] somewhere in
           its name (e.g. 'ja.train.ids' for the training file).
    """

    def __init__(self, input_dir):
        super().__init__(input_dir)

    def get_paths(self, kind):
        check_kind(kind)
        yield self.input_dir.glob(f'*{kind}*')[0]

    def num_files(self, kind):
        return 1


class FileFlatGenerator(LocalPathGenerator):
    """Files are given as flat list. No specification of train/valid/test.

    Assumptions:
        input_dir contains one or more data files (no subdirs) for which
        we need to determine the partitioning into train/valid/test ourselves.
    """

    def __init__(self,
                 input_dir: io_util.Directory,
                 data_split: Union[dict, util.DotDict] = None,
                 shuffle: bool = False,
                 seed: int = 1):
        """
        Args:
            input_dir: directory containing data files.
            data_split: dictionary of {str => float} with keys equal
                to the elements of ALLOWED_KINDS (train, valid, test).
                Constraint: sum(data_split.values()) == 1 (approx).
            shuffle: whether to shuffle the list of file paths used.
            seed: random seed used when shuffling.
        """
        if data_split is not None:
            data_split = util.DotDict(data_split)

        super().__init__(input_dir)
        # Enforce assumptions.
        if self.input_dir.has_subdirs():
            raise RuntimeError(f'Expected {self.input_dir} to contain only '
                               f'files, not subdirectories.')

        self._file_paths = self.input_dir.get_file_paths()
        if shuffle:
            # Sort first to ensure we'd always *start* with the same list.
            self._file_paths = sorted(self._file_paths)
            random.Random(seed).shuffle(self._file_paths)

        self._counter = TrainValidTestCounter(
            len(self._file_paths),
            data_split=data_split)

    def get_paths(self, kind, max_num_paths=None):
        """Retrieves all files recursively contained in the subdir `kind`."""
        begin, end = self._counter.get_range(kind)
        if max_num_paths is not None and max_num_paths < (end - begin):
            end = begin + max_num_paths
        # yield from self._file_paths[begin:end]
        for path in self._file_paths[begin:end]:
            yield path

    def num_files(self, kind):
        """
        Returns:
            number of files associated with `kind`.
        """
        return self._counter.num(kind)


class DirectorySplitGenerator(LocalPathGenerator):
    """
    Assumptions:
        1) Single subdirectory for each of ALLOWED_KINDS.
        2) Subdir for ALLOWED_KINDS[i] has name equal to the value of
           ALLOWED_KINDS[i] (e.g. 'train').
    """

    def __init__(self, input_dir):
        super().__init__(input_dir)
        # Enforce assumptions.
        if self.input_dir.get_num_subdirs(nested=False) != len(ALLOWED_KINDS):
            raise RuntimeError(
                f'Expected input_dir to contain exactly {len(ALLOWED_KINDS)} '
                f'subdirs, with names: {ALLOWED_KINDS}.')

    def get_paths(self, kind):
        """Retrieves all files recursively contained in the subdir `kind`."""
        check_kind(kind)
        # yield from self.input_dir.subdir(kind).get_file_paths(nested=True)
        for path in self.input_dir.subdir(kind).get_file_paths(nested=True):
            yield path




