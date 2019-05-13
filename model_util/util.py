# -*- coding: utf-8 -*-

"""Generally useful things. If I find myself writing the same code/functions a lot
in different files, it is likely I'll just append it here.
"""

from __future__ import print_function

import copy
import functools
import itertools
import glob
import inspect
import logging
import os
import tempfile
import time
import warnings
from copy import deepcopy
from collections import OrderedDict, defaultdict
from functools import partial
from functools import wraps
from multiprocessing import Pool, cpu_count
from operator import itemgetter
from enum import Enum
from typing import Union

import numpy as np
import yaml
from pympler.asizeof import asizeof  # profiling memory usage

import colorama
colorama.init()

LIST_TYPES = (list, tuple, np.ndarray)
STRING_TYPES = (type(b''), type(u''))
NUM_CORES = cpu_count()


def lmap(fn, iterable):
    return list(map(fn, iterable))


def styled_string(style: colorama.Style, s: str):
    return f'{style}{s}{colorama.Style.RESET_ALL}'


class Logger:
    """Custom Logger implementation.

    Motivated by the fact that if you use both the standard logging
    module and tensorflow at the same time, your logs get duplicated.
    TensorFlow does not view this as a bug, so it probably won't get
    fixed anytime soon. Hence I made this simple implementation that
    will ensure no duplicate logs.
    """

    class Level(Enum):
        TRACE = 0
        DEBUG = 1
        INFO = 2
        WARNING = 3
        ERROR = 4

    LEVEL_TO_COLOR = {
        Level.TRACE: colorama.Fore.WHITE,
        Level.DEBUG: colorama.Fore.CYAN,
        Level.INFO: colorama.Fore.GREEN,
        Level.WARNING: colorama.Fore.YELLOW,
        Level.ERROR: colorama.Fore.RED,
    }

    # Maps name prefixes to log file paths, if the user wants to
    # log to a file in addition to stdout. It is a global attribute
    # in order to mimic the standard logging package behavior, where the
    # user only needs to specify the file once, and any logger created
    # thereafter will automatically also log to that file if it has the
    # same name prefix.
    PREFIX_TO_LOG_FILE = {}
    PREFIX_TO_VERBOSITY = {}

    def __init__(self,
                 name: str,
                 verbosity: Union[Level, str] = None,
                 file_path=None):
        self.name = name
        self.prefix = name.split('.')[0]

        # Verbosity.
        if isinstance(verbosity, str):
            verbosity = Logger.Level[verbosity.upper()]

        if verbosity is not None and \
                self.prefix not in self.PREFIX_TO_VERBOSITY:
            self.PREFIX_TO_VERBOSITY[self.prefix] = verbosity

        # File path.
        if file_path is not None and self.prefix not in self.PREFIX_TO_LOG_FILE:
            self.PREFIX_TO_LOG_FILE[self.prefix] = file_path

    def trace(self, msg: str):
        if self._should_log(Logger.Level.TRACE):
            self._log(self._formatted(msg, Logger.Level.TRACE))

    def debug(self, msg: str):
        if self._should_log(Logger.Level.DEBUG):
            self._log(self._formatted(msg, Logger.Level.DEBUG))

    def info(self, msg: str):
        if self._should_log(Logger.Level.INFO):
            self._log(self._formatted(msg, Logger.Level.INFO))

    def warning(self, msg: str):
        if self._should_log(Logger.Level.WARNING):
            self._log(self._formatted(msg, Logger.Level.WARNING))

    def error(self, msg: str):
        if self._should_log(Logger.Level.ERROR):
            self._log(self._formatted(msg, Logger.Level.ERROR))

    def _should_log(self, level):
        return self.PREFIX_TO_VERBOSITY.get(
            self.prefix, Logger.Level.DEBUG).value <= level.value

    def _log(self, s: str):
        # Log to stdout.
        print(s)
        # Optionally log to file path.
        file_path = self.PREFIX_TO_LOG_FILE.get(self.prefix)
        if file_path is not None:
            with open(file_path, 'a') as f:
                print(s, file=f, flush=True)

    def _formatted(self, msg: str, level: Level):
        frame_info = inspect.stack()[2]
        caller_module = inspect.getmodule(frame_info.frame)
        if caller_module is None:
            module_name = ''
        else:
            module_name = caller_module.__name__

        function_name = frame_info.function
        level = self._colored_level(level)

        from colorama import Style
        module_name = styled_string(Style.DIM, f'[{module_name}]')
        function_name = styled_string(Style.DIM, f'[{function_name}]')
        msg = styled_string(Style.BRIGHT, msg)
        return f'{level} {module_name} {function_name} {msg}'

    def _colored_level(self, level: Level):
        return f'{self.LEVEL_TO_COLOR[level]}' \
            f'[{level.name}]' \
            f'{colorama.Style.RESET_ALL}'


def get_logger(name,
               is_root_logger=False,
               filename=None,
               console_level='info'):
    """Sets up logger to have distinct console handler and file handler, and
    allows freedom to change the log level independently for each.

    Full disclosure: I've noticed repeated log messages occur at
    (seemingly) unpredictable spots, and I'm aware I need to read
    more deeply into the logging docs to understand the reason.
    """

    def log_n(self, level, n, msg):
        if self.ctr[msg] < n:
            getattr(self, level)(msg)
        self.ctr[msg] += 1

    kwargs = {'name': name}
    if is_root_logger:
        kwargs['file_path'] = filename
        kwargs['verbosity'] = str(console_level)

    l = Logger(**kwargs)
    l.ctr = defaultdict(int)
    l.debug_n = partial(log_n, l, 'debug')
    l.info_n = partial(log_n, l, 'info')
    l.warning_n = partial(log_n, l, 'warning')
    l.error_n = partial(log_n, l, 'error')
    return l


logger = get_logger('project.util')


def timed_function(name=None):
    """(Decorator) Show how long the functions take to run.

    Args:
        name: optional function name to use when displaying the runtime. If not
        given, will try and figure out the function name itself.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):

            # We will attempt to uncover if `fn` is a H.O.F. -- if so,
            # assign the name of the function it operates on to:
            arg_fname = None

            # Extract (private) useful information for printing, if given.
            # Clients need not understand/be aware of this at all.
            if kwargs.get('_print_info') is not None:
                arg_fname = kwargs.pop('_print_info')

            # Run the wrapped function and record runtime.
            start_time = time.time()
            res = fn(*args, **kwargs)
            stop_time = time.time()

            col_widths = [12, 7]
            if len(args) == 0:
                logger.debug('{rt:<{cw[0]}} {f:<{cw[1]}}: {t:.3f} seconds.'.format(
                    rt='Runtime for:',
                    f=args[0].func,
                    cw=col_widths,
                    t=stop_time - start_time))
            else:
                if arg_fname is None:
                    if inspect.isfunction(args[0]) or hasattr(args[0], '__name__'):
                        arg_fname = args[0].__name__
                    elif isinstance(args[0], functools.partial):
                        # When functools.partial has been used:
                        arg_fname = args[0].func.__name__
                        if arg_fname == 'listable':
                            # quick n dirty hack to filter out clean.listable calls.
                            return res
                    else:
                        arg_fname = '...'

                logger.debug('{rt:<{cw[0]}} {f:<{cw[1]}}({a}): {t:.3f} seconds.'.format(
                    rt='Runtime for:',
                    cw=col_widths,
                    f=name or fn.__name__,
                    a=arg_fname,
                    t=stop_time - start_time))

            return res
        return wrapper
    return decorator


def listify(fn):
    """(Decorator) Allow `fn` to naturally handle either a single
    item or a list as first argument. It also returns the same back.
    """
    def wrapper(self, first_arg, *args, **kwargs):
        # Check that first arg is list.
        single_arg = False
        if not isinstance(first_arg, LIST_TYPES):
            single_arg = True
            first_arg = [first_arg]
        # Call the function.
        res = fn(self, first_arg, *args, **kwargs)
        # Un-listify if wasn't a list.
        if res is not None and single_arg:
            return res[0]
        return res
    return wrapper


def parallelize(fn, num_partitions=32, num_cores=NUM_CORES):
    """Sort of a decorator, but have to wrap manually due to namespace
    issues with multiprocessing.Pool. By this I mean doing
        my_func = util.parallelize(_my_func)
    Note that you can still call the function as you would normally, and
    now I've added support for arbitrary **kwargs as well, thanks to
    functools.partial.

    Since using this with partial for cleaning functions is so common, I'll
    break it down in detail using clean.replace_entities as an example.

    1. Initial assignment:
        replace_entities = util.parallelize(partial(listable, fn=_replace_entities))
    2. Within `parallelize` (here).
        - fn = clean.listable(texts, fn=_replace_entities, **kwargs)
             = <some function that accepts params: (texts, **kwargs)>
        - return wrapper(iterable, **kwargs).
            - does another partial, passing in **kwargs to the underlying function.
    """
    def wrapper(iterable, **kwargs):
        """Wrapper is what fn becomes, basically."""
        # Fill in kwargs to function we are doing the parallel map over.
        parallel_fn = partial(fn, **kwargs)
        return parallel_map(parallel_fn, iterable,
                            num_partitions=num_partitions,
                            num_cores=num_cores)
    return wrapper


@timed_function('parallel_map')
def parallel_map(fn, iterable, num_partitions=32, num_cores=NUM_CORES):
    """ Based on great explanation from
    'Pandas in Parallel' (racketracer.com).

    N.B.: fn must itself accept **iterable** arg.
    """
    iterable = np.array_split(iterable, num_partitions)
    pool = Pool(num_cores)
    res = pool.map(fn, iterable)
    try:
        iterable = np.concatenate(res)
    except ValueError:
        logger.debug('Could not concat parallel map. Extending via loop.')
        iterable = [line for chunk in res for line in chunk]
    pool.close()
    pool.join()
    return iterable


def get_glove_path(base_dir, dim=25, prompt_if_multiple_found=True):
    matches = glob.glob(f'{base_dir}/glove.*{dim}d.txt')
    if len(matches) == 0:
        raise FileNotFoundError(f'Could not find GloVe file for dimension {dim}.')
    elif len(matches) == 1 or not prompt_if_multiple_found:
        return matches[0]
    else:
        matches_str = '\n'.join(f'{i}: {m}' for i, m in enumerate(matches))
        print(f'\nMultiple GloVe files found with dim={dim}. '
              f'Enter number of choice:\n{matches_str}')
        choice = int(input('Number (default=0): ') or 0)
        print(f'Using: {os.path.basename(matches[choice])}\n')
        return matches[choice]


def read_word_vec_file(path, vocab_size=None, start_line=0):
    word2vec = {}
    with open(path) as f:
        for _ in range(start_line):
            f.readline()

        for line in f:
            word, vec = line.split(' ', 1)
            try:
                word2vec[word] = np.fromstring(vec, sep=' ')
            except Exception:
                print('word:', word)
                print('vec:', vec)
                raise ValueError

            if vocab_size is not None and len(word2vec) >= vocab_size:
                break
    return word2vec


def get_glove(base_dir, dim=25, vocab_size=None, prompt_if_multiple_found=True):
    """Load glove word2vec dictionary with vector of size `dim`.
    Args:
        base_dir: path to directory containing the individual glove files.
        dim: (int) dimensionality of word vectors.
        vocab_size: (int) Number of vectors to get. Default is to get
            all of them in the provided file.
        prompt_if_multiple_found: (bool) whether to prompt user if multiple GloVe files are
            found with the specified `dim`. If False, choose the first match.
    """
    if base_dir is None:
        raise ValueError('get_glove received base_dir=None.')
    glove_path = get_glove_path(base_dir, dim, prompt_if_multiple_found)
    if not os.path.exists(glove_path):
        raise FileNotFoundError(
            'Could not GloVe file: {}. Please go to {} and '
            'download/unzip "glove.6B.zip" to the "glove" '
            'directory.'.format(glove_path, 'https://nlp.stanford.edu/projects/glove/'))

    return read_word_vec_file(glove_path, vocab_size=vocab_size)


def force_extension(path, ext):

    """Return `path` with guaranteed extension '.ext'."""
    return '{}.{}'.format(os.path.splitext(path)[0], ext.lstrip('.'))



def get_yaml_config(path):
    path = force_extension(path, 'yml')
    with open(path) as file:
        config = yaml.load(file)
    return DotDict(config)


def merge_dicts(default_dict, preference_dict):
    """Preferentially (and recursively) merge input dictionaries.

    Ensures that all values in preference dict are used, and
    all other (i.e. unspecified) items are from default dict.

    N.B.: Notable (but sensible imo) caveat is, for conflict resolution, I will
    not overwrite default_dict[key] with preference_dict[key] IF
    preference_dict[key] is None.
    """
    if preference_dict is None:
        return default_dict

    merged_dict = copy.deepcopy(default_dict)
    for pref_key in preference_dict:
        if isinstance(preference_dict[pref_key], dict) and pref_key in merged_dict:
            # Special case: allow default_dict[k] to be None even if pref_dict[k] is dict.
            merged_dict[pref_key] = merged_dict[pref_key] or {}
            # Dictionaries are expected to have the same type structure.
            # So if any preference_dict[key] is a dict, then require that
            # default_dict[key] also be a dict (if it exists, that is).
            assert isinstance(merged_dict[pref_key], dict), \
                "Expected default_dict[%r]=%r to have type dict." % \
                (pref_key, merged_dict[pref_key])
            # Since these are both dictionaries, can just recurse.
            merged_dict[pref_key] = merge_dicts(merged_dict[pref_key],
                                                preference_dict[pref_key])
        elif pref_key in merged_dict:
            # Allow preference_dict to overwrite, as long as the value
            # can be coerced to True.
            if preference_dict[pref_key] is None:
                continue
            merged_dict[pref_key] = preference_dict[pref_key]
        else:
            # Insert the (new; no conflict) key-value pair.
            merged_dict[pref_key] = preference_dict[pref_key]
    return merged_dict


def parse_config(args=None, config_path=None, **kwargs):
    """Parse configuration dictionary from the arguments given. Depending
    on what is given as arguments to this function, the returned dictionary is
    a merged version of those in the arguments. Details below.

    Args:
        args: (obj) returned by parser.parse_args()
        config_path: (str) relative path or basename for yaml config file.
            If basename, e.g. 'imdb', assumes you want config/imdb.yml.
        kwargs: miscellaneous params to insert into the returned dictionary.

    Order of precedence for value conflict-resolution (merging):
        1. kwargs. Guaranteed to be in dictionary as given.
        2. args.
        3. config_path.
    """
    config = {}

    # Begin with finding the configuration path if given. Again,
    # we view this as the most "default" of the possibilities here.
    if config_path is not None:
        config_path = force_extension(config_path, 'yml')
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f'Config path does not exist: {config_path}.')
        config = get_yaml_config(config_path)

    # Next check contents of `args`, if given. Again,
    # we allow this to overwrite anything in path_config.
    if args is not None:
        args = vars(args)
        # First check that the known nested dictionaries are
        # in proper dictionary form (not arg strings).
        for key in ['graph', 'train', 'data']:
            if all((key in args, isinstance(args.get(key), str))):
                args[key] = yaml.load(args[key])
        config = merge_dicts(default_dict=config, preference_dict=args)

    # Finally, insert anything given by kwargs.
    if kwargs is not None:
        config = merge_dicts(default_dict=config, preference_dict=kwargs)
    return DotDict(config)


def items_recursive(dictionary, recurse_on='keys', level=1):
    for key, value in dictionary.items():
        # Primitives.
        if recurse_on == 'keys':
            yield key, level
        # elif recurse_on == 'values' and isinstance(value, (int, float, str)):
        elif recurse_on == 'values' and not isinstance(value, (dict, list)):
            yield value, level

        # Recursion.
        if isinstance(value, dict):
            for item in items_recursive(value, recurse_on=recurse_on, level=level + 1):
                yield item
        elif isinstance(value, list):
            for value_item in value:
                if isinstance(value_item, dict):
                    for item in items_recursive(value_item, recurse_on=recurse_on, level=level + 1):
                        yield item
                elif recurse_on == 'values':
                    yield value_item, level + 1


def print_box(title, content=None, print_width=None):
    """Prints a pretty box and displays `content`.

    Funny how such an ugly function can produce such nice-looking boxes.

    Args:
        title: (str) centered-title for the box.
        content: object containing content for display. Supported types:
            dict, list, list of two-tuples.

    Example: print_box('Title', {'key1': 'val1', 'key2': 'val2'}) will print:
        --------------
        |   Title    |
        | key1: val1 |
        | key2: val2 |
        --------------
    """
    # Save me from myself.
    if not isinstance(title, str) and content is None:
        raise ValueError('Sigh. You forgot that print_box requires a title again.')

    # Space taken up by non-content for each supported type(content).
    dict_padding = 6
    list_padding = 6

    if print_width is None:
        print_width = 0  # It will figure it out.

    def is_two_tuple(x):
        return isinstance(x, tuple) and len(x) == 2

    def get_longest(dictionary, recurse_on='keys'):
        """Don't try to understand this. Stop reading. Stop."""
        res = max(list(items_recursive(dictionary, recurse_on=recurse_on)),
                  key=lambda i: len(str(i[0])))
        # hackety-hack-hack quack i write code that is cwap:
        longest_list_item = 0
        if recurse_on == 'values':
            for k, v in dictionary.items():
                if isinstance(v, LIST_TYPES):
                    if is_empty(v):
                        v_longest = 0
                    else:
                        v_longest = max(v, key=lambda i: len(str(i)))

                    if len(str(v_longest)) > len(str(longest_list_item)):
                        longest_list_item = v_longest
            if len(str(longest_list_item)) > len(str(res[0])):
                res = (longest_list_item, res[1])
        return res

    # ===========================================
    # Setup: Gather info about `content`.
    # ===========================================
    if content is not None:
        # Convert lists of two-tuples to dictionaries.
        if all(lmap(is_two_tuple, content)):
            content = OrderedDict(content)
        # Determine the lengths of the longest key/values, so we can set
        # print_width large enough to contain them.
        if isinstance(content, dict):
            longest_key, level = get_longest(content, recurse_on='keys')
            longest_key = len(str(longest_key)) + (level - 1) * 4
            longest_val = len(str(get_longest(content, recurse_on='values')[0]))
            width = longest_key + longest_val + dict_padding
            leftover = max(print_width - width, 0)
            print_width = longest_key + longest_val + leftover + dict_padding
        elif hasattr(content, '__iter__'):
            longest_item = max([len(str(item)) for item in content])
            leftover = max(print_width - longest_item, 0)
            print_width = longest_item + leftover + list_padding
            longest_key = 0
        else:
            raise TypeError('Unknown content type: {}'.format(type(content)))

    # ===========================================
    # Execute: print that beautiful box.
    # ===========================================
    print_width = max(print_width, len(title) + 4)
    print('\n', '-' * print_width, sep='', flush=True)
    print('|{0:^{pw}}|'.format(title.title(), pw=print_width - 2), flush=True)
    if content is not None:
        _print_box(content, longest_key, print_width - longest_key - dict_padding)
    print('-' * print_width, end='\n\n', flush=True)


def _print_box(content, left_width, right_width, left_pad=0, is_in_list=False):
    """Helper for print_box. Recursively prints `content`."""
    left_pad_step = 3
    if isinstance(content, dict):
        for k, v in content.items():
            # Recurse on values that are themselves dicts or lists.
            if isinstance(v, (dict, list)):
                if is_in_list:
                    k = '- ' + str(k)
                print('| {lp}{k:<{lw}}: {v:<{rw}} |'.format(
                    lp=' ' * left_pad, k=str(k), v='',
                    lw=int(left_width)-left_pad, rw=right_width), flush=True)
                _print_box(v, left_width, right_width, left_pad + left_pad_step)
            else:
                print('| {lp}{k:<{lw}}: {v:<{rw}} |'.format(
                    lp=' ' * left_pad, k=str(k), v=str(v),
                    lw=int(left_width) - left_pad, rw=int(right_width)), flush=True)
    elif hasattr(content, '__iter__'):
        for item in content:
            if isinstance(item, dict):
                _print_box(item, left_width, right_width, left_pad, is_in_list=True)
            else:
                print('| {lp}- {i:<{pw}} |'.format(
                    lp=' ' * left_pad, i=str(item),
                    pw=int(left_width - left_pad + right_width)), flush=True)


def mem_stats(obj):
    """Prints memory usage of object."""
    content = {
        'Memory Usage': '{:.4f} GiB'.format(float(asizeof(obj)) / 1e9),
        'Object Type': type(obj)}
    if hasattr(obj, '__len__'):
        content['Length'] = len(obj)
    print_box(
        title='Memory Stats',
        content=content)


def swap_key_vals(the_dict):
    # Be sensitive to OrderedDict objects being passed here.
    items_swapped = [(v, k) for k, v in the_dict.items()]
    if isinstance(the_dict, OrderedDict):
        return OrderedDict(items_swapped)
    return dict(items_swapped)


def is_empty(iterable, none_ok=False):
    if none_ok:
        return iterable is None or len(iterable) == 0
    return len(iterable) == 0


def attrgetter(attr_name):
    """Same as operator,itemgetter, but uses
    __getattr__ instead of __getitem__.
    """
    def getter(item):
        return getattr(item, attr_name)
    return getter


def index_generator(shape):
    """Iterator over indices of array with specified shape in column-major
    order. For example, if shape == (2, 4, 6), this performs:
        for k in range(6):
            for j in range(4):
                for i in range(2):
                    yield i, j, k

    Args:
        shape: returned by np.ndarray.shape.

    Yields:
        indices into np.ndarray whose shape is given by `shape`.
    """

    def gen_n(n):
        assert n > 0, f'{n} is not greater than zero.'

        # Base case.
        if n == 1:
            # yield from range(shape[0])
            for x in range(shape[0]):
                yield x
        else:
            # Induction on n.
            for i in range(shape[n - 1]):
                for remaining_indices in gen_n(n=n-1):
                    if n - 1 == 1:
                        indices = (remaining_indices, i)
                    else:
                        indices = remaining_indices + (i,)
                    yield indices

    return gen_n(n=len(shape))


def save_column_major(arr, path):
    """Saves flattened out array elements in column-major ordering.

    Args:
        arr: np.ndarray whose elements we will save.
        path: text file path to save the sequence of elements.
    """
    with open(path, 'w+') as f:
        for index in index_generator(arr.shape):
            f.write(f'{arr[index]} ')


def oneof(*args):
    """
    Args:
        *args: boolean values

    Returns:
        True if exactly 1 of the values is True.
    """
    return sum(args) == 1


def one_is_not_none(*args):
    """
    Args:
        *args: boolean values

    Returns:
        True if exactly 1 of the values is not None.
    """
    return oneof(*list(map(lambda a: a is not None, args)))


def all_are_not_none(*args):
    """
    Args:
        *args: boolean values

    Returns:
        True if all args are not equal to None.
    """
    return all(map(lambda a: a is not None, args))


def pad(arr, pad_val=0, max_len=None):
    """Pads `arr` such that all elements have the same length, determined by
    the length of the longest element.

    Args:
        arr: list of sequences of possibly different lengths.
        pad_val: value to use for padding.
        max_len: max number of tokens per row of `arr`. If None,
            will use the length of the longest sequence.

    Returns:
        np.ndarray with shape (len(arr), max_len)
    """
    if max_len is None:
        max_len = len(max(arr, key=len))

    padded_arr = deepcopy(arr)
    for i in range(len(padded_arr)):
        seq_len = len(arr[i])
        if seq_len >= max_len:
            padded_arr[i] = arr[i][:max_len]
        else:
            num_pad = max_len - seq_len
            if num_pad != 0:
                padded_arr[i] = list(arr[i]) + list(itertools.repeat(
                    pad_val, num_pad))
            else:
                padded_arr[i] = arr[i]
    return np.asarray(padded_arr)


def replace(old, new, arr):
    return [item if item != old else new for item in arr]


class DotDict(dict):
    """A dict that supports indexing via my_dict.some_attr in addition to
    the more traditional my_dict['some_attr'].
    """

    def __init__(self, data=None):
        """Support arbitrary nesting of dictionaries in data, in that
        we make those nested dicts DotDict instances, too.
        """
        if data is None:
            data = {}

        super(DotDict, self).__init__(data)
        for k in data:
            # N.B.: use type(), not isinstance(). We don't want to
            # cast, e.g. defaultdict to DotDict, just primitive dicts.
            if type(data[k]) == dict:
                self[k] = DotDict(data[k])
            else:
                if isinstance(data[k], dict) and type(data[k]) != DotDict:
                    logging.warning(f'Preserving type {type(data[k])} and '
                                    f'not casting to DotDict.')
                self[k] = data[k]

    def update_recursively(self, **kwargs):
        for k, v in kwargs.items():
            self._update_recursively(k, v)
            if not self.search(k):
                self[k] = v

    def _update_recursively(self, key, value):
        for k in self:
            if k == key:
                self[k] = value
            elif isinstance(self[k], DotDict):
                # Ensure dicts are DotDicts whenever possible.
                self[k]._update_recursively(key, value)

    def search(self, key):
        for k, _ in self._flat_iter():
            if k == key:
                return True
        return False

    def count(self, key):
        return len(self.find_all(key))

    def find(self, key, default=None):
        """Returns the value assoc. with key `key`, at the
        highest level found."""
        matches = self.find_all(key)

        if len(matches) == 0:
            return default
        else:
            if len(matches) > 1:
                logging.debug(f'Found {len(matches)} entries with '
                              f'key == {key}. Only returning the first!')
            return matches[0]

    def find_all(self, key):
        """
        Returns: list of values that had their key == `key`.
        """
        return lmap(itemgetter(1),
                    filter(lambda item: item[0] == key, self._flat_iter()))

    def as_dict(self):
        res = dict(deepcopy(self))
        for k in res:
            if isinstance(res[k], DotDict):
                res[k] = dict(res[k].as_dict())
            elif isinstance(res[k], dict):
                res[k] = DotDict(res[k]).as_dict()
        return dict(res)

    def copy(self):
        return DotDict(self.as_dict())

    def _flat_iter(self):
        """Yields all (k, v) pairs, recursing on any v that are DotDicts."""
        for k, v in self.items():
            if isinstance(v, (dict, DotDict)):
                if type(v) == dict:
                    logging.warning(
                        f'Iterating by converting {type(v)} to DotDict.')

                # First yield the dictionary in full.
                yield k, v
                # Then yield recursively into the dictionary.
                # yield from DotDict(v)._flat_iter()
                for x in DotDict(v)._flat_iter():
                    yield x
            else:
                yield k, v

    def __getitem__(self, key):
        if key not in self:
            raise AttributeError('{} not in {}'.format(key, repr(self)))

        val = super(DotDict, self).__getitem__(key)
        return val

    def __setitem__(self, key, value):
        # N.B.: use type(), not isinstance(). We don't want to
        # cast, e.g. defaultdict to DotDict, just primitive dicts.
        if type(value) == dict:
            value = DotDict(value)
        super(DotDict, self).__setitem__(key, value)

    def __getattr__(self, name):
        return self.__getitem__(name)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)


def has_params(cls):
    """Class decorator for classes that utilize a DEAFULT_PARAMS dictionary.

    Use this if your class satisfies all of the following:
        - It has a class dictionary called DEFAULT_PARAMS.
        - It accepts a `params` kwarg in __init__.
        - It preferentially merges `params` into DEFAULT_PARAMS.
        - It wants to allow accessing elements of `self.params` directly, e.g.
          by doing `self.key` instead of `self.params.key`.
    """
    class Wrapper(cls):
        def __init__(self, *args, **kwargs):
            args = list(args)
            if all((len(kwargs) == 0, len(args) == 1)):
                params = args.pop(0)
            elif 'params' not in kwargs:
                raise ValueError(f'Expected `params` to be passed as kwarg '
                                 f'to {cls}. Make sure to explicitly type '
                                 f'`params=...` (i.e. kwarg format). '
                                 f'Received kwargs: {kwargs}')
            else:
                params = kwargs.pop('params')

            if hasattr(cls, 'DEFAULT_PARAMS'):
                params = DotDict(merge_dicts(
                    default_dict=cls.DEFAULT_PARAMS,
                    preference_dict=params))
            else:
                params = DotDict(params)
            self.params = params
            cls.__init__(self, *args, **kwargs, params=params)

        def __getattr__(self, item):
            if self.params.search(item):
                return self.params.find(item)
            else:
                raise AttributeError(f'{cls} does not have attribute: {item}')

        def __setattr__(self, key, value):
            if key == 'params':
                object.__setattr__(self, key, value)
            elif self.params.search(key):
                self.params.update_recursively(**{key: value})
            else:
                super().__setattr__(key, value)

    return Wrapper


def save_yaml_config(data, path):
    data = deepcopy(data)
    if isinstance(data, DotDict):
        data = data.as_dict()

    path = force_extension(path, 'yml')
    with open(path, 'w+') as file:
        yaml.dump(data, file, default_flow_style=False)


def dotted_string(obj, max_num_chars=50):
    res = str(obj)
    if len(res) <= max_num_chars:
        return res
    else:
        return res[:max_num_chars - 3] + '...'


def chainmap(*fns, iterable=None):
    """Applies sequence of functions to an iterable.

    For example, the following are equivalent:
        map(f, map(g, map(h, iterable)))
        chainmap(f, g, h, iterable=iterable)
    """
    if iterable is None:
        raise ValueError('You must pass iterable as a kwarg.')

    fns = list(fns)
    if len(fns) == 1:
        # yield from map(fns[0], iterable)
        for x in map(fns[0], iterable):
            yield x
    else:
        # yield from map(fns[0], chainmap(*fns[1:], iterable=iterable))
        for x in map(fns[0], chainmap(*fns[1:], iterable=iterable)):
            yield x


def has_arg(fn, arg: str):
    return arg in inspect.signature(fn).parameters


def is_running_on_bolt() -> bool:
    import turibolt as bolt
    task_id = bolt.get_current_task_id()
    return task_id is not None




class BidirectionalDict(dict):
    """Dictionary that supports lookup from key->value and value->key.

    Based on: https://stackoverflow.com/a/21894086

    N.B.: Currently enforcing that all values are unique, i.e. that
        len(self.values()) == len(set(self.values()))
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key)

    def update(self, other=None, **kwargs):
        for k, v in other.items():
            self[k] = v
        for k in kwargs:
            self[k] = kwargs[k]

    def get_forward(self, key, default=None):
        return self.get(key, default)

    def get_reverse(self, value, default=None):
        if value in self.inverse:
            assert len(self.inverse[value]) == 1
        return self.inverse.get(value, [default])[0]

    def has_key(self, key):
        return key in self

    def has_value(self, value):
        return value in self.values()

    def pop(self, k):
        val = super().pop(k)
        keys = self.inverse.pop(val)
        assert len(keys) == 1
        return val

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super().__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)

    def __delitem__(self, key):
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super().__delitem__(key)

