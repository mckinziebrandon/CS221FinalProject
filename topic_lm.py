import gensim
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np

from tensorflow.python.estimator.estimator_lib import ModeKeys

# --- IMPORTS REDACTED ---
logger = util.get_logger('montrealtools.topic_lm')


@util.has_params
class Trainer:
    DEFAULT_PARAMS = util.DotDict({
        'train_params': {
            'batch_size': None,
            'num_epochs': 1,
            'learning_rate': 0.001,
            'steps_per_print': 1,
            'steps_per_valid': None
        }, 'eval_params': {
            'steps': None,
        }, 'early_stopping': {
            'monitor': 'mean_loss',
            'min_delta': 0.0004,
            'patience': 3
        }})

    def __init__(self, keras_model, model_dir, params=None):
        assert tf.executing_eagerly()

        self.params = util.DotDict(params)
        assert isinstance(self.params, util.DotDict)
        self.model = keras_model
        self.params.print('Trainer')

        # Metrics.
        self.metrics = util.DotDict()
        for kind in ['train', 'valid']:
            self.metrics[kind] = util.DotDict({
                'mean_loss': tf.keras.metrics.Mean(),
                'acc': tf.keras.metrics.SparseCategoricalAccuracy()})

        # Callbacks.
        self.callbacks = [
            LoggingCallback(self.steps_per_print),
            BoltMetricCallback(),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: self._reset_metrics()),
            tf.keras.callbacks.EarlyStopping(**self.early_stopping),
            tf.keras.callbacks.ModelCheckpoint(
                model_dir.join('weights_{epoch}.hdf5'),
                monitor='mean_loss',
                save_best_only=True,
                save_weights_only=True)]
        self._validation_trigger = IntervalCounter(self.steps_per_valid)

    def train_and_evaluate(self, train_input_fn, valid_input_fn, callbacks=None):
        callbacks = callbacks or []
        self.callbacks.extend(callbacks)
        self._run_callbacks('set_model', self.model)
        self.model.stop_training = False

        self._run_callbacks('on_train_begin', self._logs('train'))
        for epoch in range(self.num_epochs):
            # TODO: consider whether we should send train or valid logs:
            self._run_callbacks('on_epoch_begin', epoch, self._logs('train'))
            if self.model.stop_training:
                logger.warning('Early stopping triggered.')
                break

            # Execute one full epoch.
            self._train_eval_epoch(epoch, train_input_fn, valid_input_fn)
            if self.model.stop_training:
                logger.warning('Early stopping triggered.')
                return

            self.evaluate(valid_input_fn)
            if self.model.stop_training:
                logger.warning('Early stopping triggered.')
                return

            # TODO: consider whether we should send train or valid logs:
            self._run_callbacks('on_epoch_end', epoch, self._logs('train'))
        self._run_callbacks('on_train_end', self._logs('train'))

    def evaluate(self, valid_input_fn):
        logger.trace('Starting evaluation.')
        j = 0
        for j, (x_val, y_val) in enumerate(valid_input_fn()):
            if self.eval_params.steps is not None and \
                    j == self.eval_params.steps:
                logger.trace(f'Breaking after {j} eval steps.')
                break
            if j % 20 == 0:
                logger.info(f'Eval step {j} . . .')
            self._validation_step(x_val, y_val)
        if j == 0:
            logger.error('Validation input_fn() yielded no data.')

        validation_logs = self._logs('valid')
        for cb in self.callbacks:
            # TODO: remove hasattr() check once we move to TF2.0.
            if hasattr(cb, 'on_test_end'):
                cb.on_test_end(logs=validation_logs)

    def _train_eval_epoch(self, epoch, train_input_fn, valid_input_fn):
        logger.info(f'Starting epoch {epoch}.')
        for i, (x_train, y_train) in enumerate(train_input_fn()):
            self._run_callbacks('on_train_batch_begin', i, self._logs('train'))
            self._train_step(x_train, y_train)
            self._run_callbacks('on_train_batch_end', i, self._logs('train'))

            if self._validation_trigger.ready():
                self.evaluate(valid_input_fn)
                if self.model.stop_training:
                    return
            self._validation_trigger.next()

    def _train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.model.loss(y, logits, mask_zero=True)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))
        self.metrics.train.mean_loss(loss)
        self.metrics.train.acc(tf.expand_dims(y, axis=-1), logits)
        return loss

    def _validation_step(self, x, y):
        logits = self.model(x, training=False)
        logger.warning_n(1, 'Assuming loss fn is losses.sequence_loss.')
        loss = self.model.loss(y, logits, mask_zero=True)
        self.metrics.valid.mean_loss(loss)
        sample_weight = tf.cast(tf.not_equal(y, 0), tf.int32)
        self.metrics.valid.acc(
            tf.expand_dims(y, axis=-1), logits,
            sample_weight=sample_weight)
        return loss

    def _logs(self, kind):
        return self.metrics[kind].map_value(lambda m: m.result())

    def _run_callbacks(self, method_name, *args):
        for cb in self.callbacks:
            getattr(cb, method_name)(*args)

    def _reset_metrics(self):
        for k in self.metrics:
            for m in self.metrics[k].values():
                m.reset_states()


class TopicLanguageModel(tf.keras.Model):

    def __init__(self,
                 iris_config: util.DotDict,
                 ldamodel_path: str,
                 vocab_path: str):
        assert tf.executing_eagerly()
        assert iris_config.find('embed_size') == iris_config.find('state_size')
        assert iris_config.find('num_layers') == 1
        super().__init__(self)

        self.params = iris_config
        self.vocab = Vocabulary(vocab_file=vocab_path, unk_token='UNK')
        self.model_dir = io_util.Directory('/tmp/model_dir')
        if bolt_util.i_am_a_running_bolt_task():
            self.model_dir = bolt_util.artifact_dir()

        # Topic model.
        self.ldamodel = gensim.models.ldamodel.LdaModel.load(
            ldamodel_path, mmap='r')

        # Language model.
        self.l_embedding = tf.keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embed_size,
            mask_zero=True)
        lstm_cell = tf.nn.rnn_cell.LSTMCell(
            num_units=self.state_size,
            forget_bias=0.0),
        rnn_kwargs = {'return_sequences': True}
        self.l_lstm = tf.keras.layers.RNN(cell=lstm_cell, **rnn_kwargs)
        self.l_dense = tf.keras.layers.Dense(self.vocab_size)
        self.compile(
            tf.train.AdamOptimizer(),
            loss=losses.sequence_loss,
            run_eagerly=True)
        self.current_topic_features = None

    def build_and_load(self, iris_weights):
        if self.current_topic_features is None:
            raise RuntimeError(
                'You must set current_topic_features before calling '
                'build_and_load(), because Keras\' build() method will '
                'ultimately invoke self.call().')
        self.build(input_shape=(None, None))
        self.get_layer('embedding').set_weights([
            iris_weights.embedder_embed_tensor])
        topic_weights = tf.get_variable(
            name='topic_weights',
            shape=(self.ldamodel.num_topics, 4 * self.state_size),
            initializer=tf.initializers.glorot_uniform())
        lstm_kernel = np.concatenate([
            topic_weights.numpy(),
            iris_weights['rnn0_rnn_lstm_cell_kernel']])
        self.get_layer('rnn').set_weights([
            lstm_kernel,
            iris_weights['rnn0_rnn_lstm_cell_bias']])
        self.get_layer('dense').set_weights([
            iris_weights.logits_dense_kernel,
            iris_weights.logits_dense_bias])

    def set_topic_features(self, inputs):
        tokens = np.array([
            [self.lm_id_to_token(id) for id in seq]
            for seq in inputs], dtype=np.str)
        self.current_topic_features = np.array([
            self.seq_to_topic_probs(seq) for seq in tokens
        ], dtype=np.float32)

    def __call__(self, *args, **kwargs):
        self.set_topic_features(args[0])
        return super().__call__(*args, **kwargs)

    def call(self, inputs, training=None, mask=None):
        if isinstance(inputs, list):
            # Keras will insert inputs into a list because I HAVE NO IDEA
            util.assert_equal(1, len(inputs))
            inputs = inputs[0]

        inputs = self._to_int64(inputs)
        # if tf.rank(inputs).numpy() != 2:
        #     inputs = tf.reshape(inputs, [self.batch_size, -1])

        topic_probs = self.current_topic_features
        word_embeddings = self.l_embedding(inputs)
        lstm_outputs = self.l_lstm(
            tf.concat((topic_probs, word_embeddings), axis=-1),
            training=training, mask=mask)
        logits = self.l_dense(lstm_outputs)
        return logits

    def lm_id_to_token(self, lm_id):
        from tensorflow.python.framework.ops import EagerTensor
        if isinstance(lm_id, EagerTensor):
            lm_id = lm_id.numpy()
        return self.vocab.id_to_token(lm_id)

    def topic_probs(self, token) -> np.ndarray:
        if util.is_empty(self.ldamodel.id2word.doc2bow([token])):
            # Return uniform distribution if topic model doesn't know word.
            return np.array([
                1. / self.ldamodel.num_topics
                for _ in range(self.ldamodel.num_topics)
            ], dtype=np.float32)
        else:
            word_id = self.ldamodel.id2word.doc2bow([token])[0][0]
        res = np.array([
            self.ldamodel.expElogbeta[topic_id][word_id]
            for topic_id in range(self.ldamodel.num_topics)], dtype=np.float32)
        res /= np.linalg.norm(res)
        return res

    def seq_to_topic_probs(self, tokens) -> np.ndarray:
        return np.array([self.topic_probs(t) for t in tokens], dtype=np.float32)

    def train_input_fn(self, filenames):
        return self._input_fn(filenames, max_samples=self.max_samples)

    def valid_input_fn(self, filenames, batch_size=None):
        return self._input_fn(filenames, batch_size=batch_size)

    def _input_fn(self, filenames, max_samples=None, batch_size=None):
        def map_func(x):
            return tf_util.parse_text_line(
                x, self.vocab_size, self.max_seq_len, 3)
        filenames = list(filenames)
        batch_size = batch_size or self.batch_size
        dataset = tf.data.TextLineDataset(filenames)\
            .prefetch(buffer_size=None)\
            .shuffle(buffer_size=100)\
            .map(map_func=map_func)\
            .take(max_samples or -1)\
            .padded_batch(batch_size, padded_shapes=(
                tf.TensorShape([self.max_seq_len]),
                tf.TensorShape([self.max_seq_len])))
        return dataset

    def train_and_evaluate(self, path_generator, **kwargs):
        trainer = Trainer(
            keras_model=self,
            model_dir=self.model_dir,
            params=self.params.subdict('train_params', 'eval_params'))
        try:
            trainer.train_and_evaluate(
                train_input_fn=lambda: self.train_input_fn(
                    path_generator.get_paths('train')),
                valid_input_fn=lambda: self.valid_input_fn(
                    path_generator.get_paths('valid')))
        except SystemExit as e:
            logger.warning(f'Training terminated early: {e}')
        except KeyboardInterrupt:
            logger.warning('KeyboardInterrupt. Training stopped.')
        self.save_weights('weights_from_keras_lm.hdf5')

    def _to_int64(self, x):
        # WHY DO YOU WANT ME TO SUFFER
        if isinstance(x, list):
            x = np.asarray(x)
        shape_before_tf_ruins_it_for_some_reason = x.shape
        inputs = tf.to_int64(x)
        if str(inputs.shape) != str(shape_before_tf_ruins_it_for_some_reason):
            x = tf.reshape(inputs, shape_before_tf_ruins_it_for_some_reason)
        return x

    def __setattr__(self, name, value):
        """Dear Keras: I hate you.

        tf.keras.Model.__setattr__ does a bunch of wrapping and life-ruining,
        among which are converting any value for which isinstance(value, dict)
        is True to _DictWrapper(value) and thereby invalidaing any DotDict
        objects (e.g. self.params).
        """
        if isinstance(value, util.DotDict):
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, item):
        """Dear Keras: I hate you."""
        if item == 'params':
            return object.__getattribute__(self, 'params')
        elif self.params.search(item):
            return self.params.find(item)
        else:
            return super().__getattr__(item)

