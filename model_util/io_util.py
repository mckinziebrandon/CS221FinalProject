"""Utilities for interacting with the filesystem.

I'd recommend using the Directory class (end of file) as your go-to interface whenever
possible, but I've exposed most of its functionality as standalone functions if
that better suits your use-case.
"""

import os
import shutil
import glob

from . import util

logger = util.get_logger('project.model_util.io_util')


def get_filepaths(dir_path, nested=True):
    """Returns list of [full] path_generator to FILES located within `dir_path` directory."""
    if not os.path.exists(dir_path):
        raise NotADirectoryError('`dir_path` does not exist: {}'.format(dir_path))
    if not nested:
        contents = [os.path.join(dir_path, c) for c in os.listdir(dir_path)]
        return [fp for fp in contents if os.path.isfile(fp)]

    filepaths = []
    for root, dirs, files in os.walk(dir_path):
        files = [os.path.join(root, f) for f in files]
        filepaths.extend(files)
    return filepaths


def num_files_nested(dir_path):
    if not os.path.exists(dir_path):
        raise NotADirectoryError('`dir_path` does not exist: {}'.format(dir_path))
    return len(get_filepaths(dir_path, nested=True))


def get_subdirs(dir_path, nested=False):
    """Returns list of [full] path_generator to dirs located within `dir_path` directory."""
    if not os.path.exists(dir_path):
        raise NotADirectoryError('`dir_path` does not exist: {}'.format(dir_path))
    if not nested:
        subdirs = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
        subdirs = [item for item in subdirs if os.path.isdir(item)]
        return subdirs
    dirpaths = []
    for root, dirs, files in os.walk(dir_path):
        logger.debug('Appending subdir: {}'.format(root))
        dirpaths.append(root)
    return dirpaths


def dirs_with_at_least_n(root_dir, nested=False, n=0):
    """Returns subdirectories of `root_dir` that have at least `n` [possibly nested]
    files within them.
    """
    subdirs = get_subdirs(root_dir, nested=nested)

    if n == 0:
        return subdirs

    filtered_subdirs = []
    for subdir in subdirs:
        num_files = num_files_nested(subdir)
        if num_files >= n:
            filtered_subdirs.append(subdir)
        else:
            logger.debug('Ignoring: {:<40} num files: {}'.format(
                os.path.basename(subdir), num_files))
    return filtered_subdirs


class Directory:
    """Handy wrapper around the above functions for fixed `dir_path`."""

    def __init__(self, path):
        if isinstance(path, Directory):
            path = path.path
        self._path = path

    @staticmethod
    def of_file(file_path):
        """Useful creator method when you want the directory of
        the current file, e.g.:
            d = Directory.of_file(__file__)
        instead of having to type:
            d = Directory(os.path.realpath(os.path.dirname(__file__)))

        Returns:
            Full path to directory containing `file_path`.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File path not found: {file_path}')
        return Directory(os.path.realpath(os.path.dirname(file_path)))

    @property
    def path(self):
        if self._path.startswith('/'):
            return self._path
        else:
            return os.path.realpath(self._path)

    @property
    def name(self):
        return os.path.basename(self.path)

    def create(self):
        os.makedirs(self.path, exist_ok=True)

    def parent_dir(self):
        return Directory(os.path.realpath(self.join(os.pardir)))

    def delete_contents(self, delete_subdirs=True):
        """
        Args:
            delete_subdirs: if True, also recursively deletes subdirectories.
                Otherwise, only delete files in immediate directory.
        """
        # Delete files immediately beneath self.
        util.lmap(lambda f: os.unlink(f), self.get_file_paths())
        # Delete subdirs if requested.
        if delete_subdirs:
            util.lmap(lambda d: shutil.rmtree(d.path), self.get_subdirs())

    def delete(self):
        shutil.rmtree(self.path)
        assert not self.exists()

    def empty(self):
        """
        Returns: True if there are no files or directories in self.
        """
        return all((util.is_empty(self.get_file_paths()),
                    util.is_empty(self.get_subdirs())))

    def exists(self):
        return os.path.exists(self._path)

    def has_files(self):
        return len(self.get_file_paths(nested=False)) > 0

    def has_subdirs(self):
        return self.get_num_subdirs() > 0

    def has_subdir(self, name):
        return all((os.path.exists(self.join(name)),
                    os.path.isdir(self.join(name))))

    def has_file(self, name):
        return all((os.path.exists(self.join(name)),
                    os.path.isfile(self.join(name))))

    def get_file_path(self, name, require_exists=True):
        if require_exists and not self.has_file(name):
            raise FileNotFoundError(
                'File {} not found.'.format(self.join(name)))
        return self.join(name)

    def get_file_paths(self, nested=False):
        """
        Arguments:
            nested: (optional) whether to search arbitrarily deep for filepaths.
        """
        return get_filepaths(dir_path=self.path, nested=nested)

    def glob(self, pattern):
        matches = glob.glob(f'{self.path}/{pattern}')
        if len(matches) == 0:
            raise FileNotFoundError(f'No files matching glob pattern: {pattern}')
        return matches

    def subdir(self, *names, create=False, require_exists=True):
        if create:
            os.makedirs(self.join(*names), exist_ok=True)

        # Recursively check each subdir, so that it fails as early as possible
        # and the exception is actually useful. For example, if `foo` does not
        # exist, and we call subdir('foo', 'bar') we'd prefer to see
        # "foo does not exist" instead of "foo/bar does not exist".
        if require_exists and not self.has_subdir(names[0]):
            raise NotADirectoryError(
                'Subdirectory {} does not exist.'.format(self.join(names[0])))

        if len(names) == 1:     # Base case.
            return Directory(self.join(names[0]))
        else:                   # Inductive case.
            return Directory(self.join(names[0])).subdir(
                *names[1:], require_exists=require_exists)

    def get_subdirs(self, nested=False, min_total_files=0):
        raw_paths = dirs_with_at_least_n(
            self.path, nested=nested, n=min_total_files)
        return [Directory(p) for p in raw_paths]

    def get_num_subdirs(self, nested=False):
        return len(self.get_subdirs(nested=nested))

    def get_num_files(self, nested=False):
        return len(self.get_file_paths(nested=nested))

    def ls(self):
        return util.lmap(lambda d: d.path, self.get_subdirs()) + \
               self.get_file_paths()

    def join(self, *args):
        """Wraps os.path.join, where path begins with self.path.

        Returns:
            string path.
        """
        return os.path.join(self.path, *args)

    def __repr__(self):
        return self.path
