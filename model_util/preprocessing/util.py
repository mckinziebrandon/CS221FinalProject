import math
from .. import util
from typing import Union

logger = util.get_logger('project.model_util.preprocessing')

# We say that we operate on the following few "kinds" of data. This
# becomes directly baked into naming conventions.
ALLOWED_KINDS = ['train', 'valid', 'test']


def check_kind(kind):
    if kind not in ALLOWED_KINDS:
        raise ValueError(f'Expected `kind` to be one of {ALLOWED_KINDS}, '
                         f'not: {kind}')


class TrainValidTestCounter:

    def __init__(self,
                 num_total,
                 data_split: Union[dict, util.DotDict] = None,
                 percent_train=None):
        if util.all_are_not_none(data_split, percent_train):
            raise ValueError('You specified both data_split and percent_train. '
                             'Only specify data_split, as percent_train is '
                             'now deprecated.')
        if percent_train is not None:
            logger.warning('DEPRECATED: use data_split instead of '
                           'percent_train.')
            data_split = util.DotDict()
            data_split.train = percent_train
            data_split.valid = (1. - percent_train) / 2
            data_split.test = (1. - percent_train) / 2

        # Default to 90/5/5 split.
        if data_split is None:
            data_split = util.DotDict()
            data_split.train = 0.9
            data_split.valid = 0.05
            data_split.test = 0.05
        if not 0.99 <= sum(data_split.values()) <= 1.1:
            raise ValueError(f'Values of data_split must sum to one. Found '
                             f'given sum to {sum(data_split.values())}.')

        self.num_total = num_total
        self.data_split = data_split
        util.print_box('DS', data_split)

    def num(self, kind):
        check_kind(kind)
        if kind == 'train':
            return int(self.data_split.train * self.num_total)
        elif kind == 'valid':
            return math.ceil(self.data_split.valid * self.num_total)
        elif kind == 'test':
            return self.num_total - self.num('train') - self.num('valid')

    def get_range(self, kind):
        check_kind(kind)
        if kind == 'train':
            begin = 0
        elif kind == 'valid':
            begin = self.num('train')
        elif kind == 'test':
            begin = self.num('train') + self.num('valid')

        end = begin + self.num(kind)
        return begin, end
