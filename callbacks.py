import inspect
import tensorflow as tf
import turibolt as bolt

from model_util import util, io_util
from tensorflow.python import keras

logger = util.get_logger('project')


class IntervalCounter:
    """Helper class that we use for numbers representing some interval for
    which an operation is executed (at the end of the interval).
    """

    def __init__(self, interval: int, include_zero=False):
        self.interval = interval
        self.include_zero = include_zero
        self._counter = 0

    def reset(self):
        self._counter = 0

    def next(self, num_steps=1):
        if self.interval is None:
            return

        assert num_steps > 0, num_steps
        if self.interval % num_steps != 0:
            raise ValueError(
                f'IntervalCounter.next called with num_steps={num_steps}, '
                f'which is not a divisor of {self.interval}.')
        self._counter += num_steps

    def ready(self):
        if self.interval is None:
            return False

        res = self._counter % self.interval == 0
        if not self.include_zero:
            res = res and self._counter != 0
        return res


class LoggingCallback(keras.callbacks.Callback):

    def __init__(self, steps_per_print=1, level='debug'):
        super().__init__()
        self.steps_per_print = IntervalCounter(steps_per_print)
        self.epochs = util.DotDict({'test': 0})
        self.level = level

    def on_train_batch_end(self, batch, logs=None):
        if self.steps_per_print.ready():
            logs = logs or {}
            log_messages = [
                f'{name}={val:.5f}' for name, val in logs.items()]
            log_messages = ', '.join(log_messages)
            getattr(logger, self.level)(f'batch={batch}, {log_messages}')
        self.steps_per_print.next()

    def on_test_end(self, logs=None):
        logs = logs or {}
        util.print_box(f'Test {self.epochs.test} Results', {
            k: f'{v:.5f}' for k, v in logs.items()})
        self.epochs.test += 1


class BoltMetricCallback(keras.callbacks.Callback):

    def __init__(self):
        super().__init__()
        self.iteration = util.DotDict({
            'train': 0,
            'test': 0,
        })

    def on_test_end(self, logs=None):
        if not util.is_running_on_bolt():
            return
        logs = logs or {}
        bolt.send_metrics(
            {f'test_{k}': f'{v:.5f}' for k, v in logs.items()},
            iteration=self.iteration.test,
            report_as_parent=True)
        self.iteration.test += 1

    def on_train_batch_end(self, batch, logs=None):
        if not util.is_running_on_bolt():
            return
        bolt.send_metrics(
            {f'train_{k}': f'{v:.5f}' for k, v in logs.items()},
            iteration=self.iteration.train,
            report_as_parent=True)
        self.iteration.train += 1

