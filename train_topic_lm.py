#!/usr/bin/env python3.6

import os
import argparse
import numpy as np
import tensorflow as tf

from typing import List

# ------- IMPORTS REDACTED --------

from topic_lm import TopicLanguageModel, Trainer

tf.enable_eager_execution()
assert tf.executing_eagerly()

parser = argparse.ArgumentParser()
parser.add_argument(
    'stuff_dir',
    help='location where iris_to_weights.hdf5.py did everything.')
parser.add_argument(
    '-s', '--source', default='fox')
parser.add_argument(
    '-m', '--mode', default='train', help='train, predict, eval, or sample')
parser.add_argument(
    '-w', '--weights_path', default='weights.hdf5')
parser.add_argument(
    '-l', '--ramble_length', default=10, type=int)
parser.add_argument(
    '--seeds', type=str, default='breaking news from the border of Mexico')
# '--seeds', type=str, default='I can\'t believe she would say that about me')
parser.add_argument(
    '-k', '--top_k', type=int, default=2)
parser.add_argument(
    '-t', '--temperature', type=float, default=1)
parser.add_argument(
    '--params',
    metavar='KEY=VALUE', nargs='+', default=None,
    help='Set a number of key-value pairs '
         '(do not put spaces before or after the = sign). '
         'If a value contains spaces, you should define '
         'it with double quotes: '
         'foo="this is a sentence". Note that '
         'values are always treated as strings.')
args = parser.parse_args()


def is_castable(s: str, type):
    try:
        type(s)
    except ValueError:
        return False
    return True


def maybe_convert_to_number(s: str):
    if is_castable(s, int):
        return int(s)
    elif is_castable(s, float):
        return float(s)
    else:
        return s


def parse_key_val_pair(s: str):
    key, val = s.split('=', maxsplit=1)
    return key.strip(), maybe_convert_to_number(val)


def parse_key_val_pairs(items: List[str]) -> util.DotDict:
    return util.DotDict({key: val for key, val in map(parse_key_val_pair, items)})


stuff_dir = io_util.Directory(args.stuff_dir)
assert stuff_dir.exists(), stuff_dir
iris_dir = stuff_dir.subdir('iris_to_weights_hdf5')

config = util.parse_config(config_path=stuff_dir.join('config.yml'))
assert isinstance(config, util.DotDict), type(config)

sampling_config = util.DotDict({
    'temperature': args.temperature,
    'top_k': args.top_k,
    'ramble_length': args.ramble_length})
log_filename = '_'.join([f'{k}:{v}' for k, v in sampling_config.items()])
log_filename = f'/tmp/{args.source}_logs/{log_filename}.log'
if bolt_util.i_am_a_running_bolt_task():
    log_filename = bolt_util.log_dir().join('train_topic_lm.log')
logger = util.get_logger(
    'montrealtools',
    filename=log_filename,
    is_root_logger=True,
    console_level='debug')


if args.params is not None:
    args.params = parse_key_val_pairs(args.params)
    config.update_recursively(**args.params)

batch_size = config.find('batch_size')


class FakeVectorizerForMontrealIDs:

    def __init__(self):
        from montrealtools.model_util.preprocessing import reader
        self.vocab = reader.Vocabulary(
            vocab_file='/Users/brandon/LocalDocuments/models/english_language_model_vocabulary.txt',
            unk_token='UNK')

    def text_to_vector(self, text: str):
        tokens = text.split()
        return [self.vocab.token_to_id(t)
                or self.vocab.token_to_id(self.vocab.unk_token)
                for t in tokens]

    def sequences_to_docs(self, stupid_docs):
        assert stupid_docs.shape[1] == 1 and len(stupid_docs.shape) == 3, stupid_docs.shape
        pred_ids = stupid_docs[:, 0]
        return [
            ' '.join([self.vocab.id_to_token(i) for i in doc])
            for doc in pred_ids
        ]

    def sent_tokenize(self, x):
        import nltk
        return nltk.sent_tokenize(x, language='english')


def logits_to_preds(logits):
    logits = tf.convert_to_tensor(logits)
    assert isinstance(logits.shape, tf.TensorShape)
    probs = tf.keras.activations.softmax(logits)
    preds = tf.argmax(probs, axis=-1)
    return preds.numpy()


def basename(filepath):
    filename = os.path.basename(filepath)
    res, _ = os.path.splitext(filename)
    return res


def load_arrays():
    """Load numpy arrays from disk."""
    inputs = np.load(iris_dir.get_file_path('inputs.npy'))
    inputs = inputs[:config.find('batch_size')]
    inputs = tf.convert_to_tensor(inputs, dtype=tf.int64)

    iris_weights = util.DotDict({
        basename(path): np.load(path)
        for path in iris_dir.subdir('numpy_weights').get_file_paths()})
    return inputs, iris_weights


def assert_equal_to_arrs(expected: List[np.ndarray], actual: List[np.ndarray]):
    util.assert_equal(len(expected), len(actual))
    for exp, act in zip(expected, actual):
        util.assert_equal(exp.shape, act.shape)
        assert np.allclose(exp, act, atol=1e-5), np.abs(exp - act).max()


def assert_equal_to_vars(expected: List[np.ndarray], actual: List[tf.Variable]):
    assert_equal_to_arrs(expected, [v.numpy() for v in actual])


def train(topic_lm: TopicLanguageModel):
    logger.info('Executing.')
    # Iris numpy arrays.
    inputs, iris_weights = load_arrays()
    # Build & load.
    topic_lm.set_topic_features(inputs)
    topic_lm.build_and_load(iris_weights)
    # Train.
    data_dir = montrealtools_dir.parent_dir().join(f'{args.source}_mrl')
    pg = FileSplitGenerator(data_dir, strict=False)
    topic_lm.train_and_evaluate(pg)


def top_k_logits(logits):
    k = args.top_k
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        # (B, V)
        min_values = values[:, -1, tf.newaxis]
        res = tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits)
        unk_index = 3
        res = tf.concat([
            res[:, :unk_index],
            tf.zeros_like(res[:, unk_index:unk_index+1]),
            res[:, unk_index+1:]
        ], axis=-1)
        # (B, V)
        return res
    return tf.cond(tf.equal(k, 0), lambda: logits, lambda: _top_k())


def sample_sequence(model):
    # context = tf.fill([batch_size, 1], start_token)
    vectorizer = FakeVectorizerForMontrealIDs()
    context = vectorizer.text_to_vector(args.seeds)
    context = np.array([context for _ in range(batch_size)])
    logger.warning('context.shape', context.shape)

    with tf.name_scope('sample_sequence'):
        def body(prev, output):
            logits = model(output)
            logits = logits[:, -1, :] / tf.to_float(args.temperature)
            logits = top_k_logits(logits)
            samples = tf.multinomial(
                logits, num_samples=1, output_dtype=tf.int32)
            return samples, tf.concat([output, samples], axis=1)

        prev, output = body(context, context)

        _, tokens = tf.while_loop(
            cond=lambda p, o: True,
            body=body,
            maximum_iterations=args.ramble_length - 1,
            loop_vars=[prev, output],
            shape_invariants=[
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None])],
            back_prop=False)
        logger.warning('tokens.shape', tokens.shape)
        return tokens


def interactive_prompt(model: TopicLanguageModel):
    logger.info('Executing.')

    def get_input():
        return input('Enter seed word(s) [Ctrl-C to quit]: ').lower()

    def feed_tokens(token_ids: List[int]) -> int:
        # (B=1, T, V)
        logits = model([token_ids])
        # (B=1, T, V)
        probs = tf.keras.activations.softmax(logits)
        # (T, V)
        probs = tf.squeeze(probs, axis=[0])
        # (T=1,)
        pred_id = probs[-1].numpy().argmax()
        logger.error('PRED ID', pred_id)
        return pred_id

    vectorizer = FakeVectorizerForMontrealIDs()
    num_iters = 0
    try:
        while True:
            model.reset_states()

            # First, feed user seed tokens through.
            if num_iters == 0 and args.seeds is not None:
                inp = args.seeds
            else:
                inp = get_input()
            pred_ids = []
            for token_id in vectorizer.text_to_vector(inp):
                pred_ids.append(token_id)
                pred_id = feed_tokens(pred_ids + [token_id])
            pred_ids.append(pred_id)

            # Predict args.ramble_length additional tokens.
            for t in range(args.ramble_length):
                pred_ids.append(feed_tokens(pred_ids))

            # Convert predictions back to natural language.
            pred_words = vectorizer.sequences_to_docs([[pred_ids]])
            assert len(pred_words) == 1, pred_words
            pred_words = pred_words[0]

            # Display generated text to user.
            pred_words = ' '.join([
                sent.capitalize()
                for sent in vectorizer.sent_tokenize(pred_words)])
            print(pred_words)
            num_iters += 1
    except KeyboardInterrupt:
        print('Ok bye.')


def ids_to_str(ids_batch: np.ndarray):
    vectorizer = FakeVectorizerForMontrealIDs()
    # Treat each batch elem as a "doc".
    ids_batch = np.array([
        [seq] for seq in ids_batch
    ])
    logger.error('docs.shape', ids_batch.shape)
    # Convert predictions back to natural language.
    pred_words = vectorizer.sequences_to_docs(ids_batch)
    assert len(pred_words) == batch_size, pred_words

    res = []
    for doc in pred_words:
        res.append(' '.join([
            sent.capitalize()
            for sent in vectorizer.sent_tokenize(doc)]))

    for i, text_sample in enumerate(pred_words):
        logger.info(f'\n-------------------- Sample {i} -------------------- '
                    f'\n{text_sample}')


def evaluate(topic_lm: TopicLanguageModel):
    data_dir = montrealtools_dir.parent_dir().join(f'{args.source}_mrl')
    pg = FileSplitGenerator(data_dir, strict=False)

    # for x, y in topic_lm.valid_input_fn(pg.get_paths('valid'), batch_size=1):
    trainer = Trainer(
        keras_model=topic_lm,
        model_dir=topic_lm.model_dir,
        params=topic_lm.params.subdict('train_params', 'eval_params'))
    trainer._run_callbacks('set_model', trainer.model)

    util.print_box('Evaluate on valid')
    trainer.evaluate(
        lambda: topic_lm.valid_input_fn(
            pg.get_paths('valid'), batch_size=1))
    loss = trainer._logs('valid')['mean_loss']
    ppl = tf.exp(loss)
    logger.error('loss', loss.numpy(), 'ppl', ppl.numpy())
    trainer._reset_metrics()
    logger.warning(f'Should be zeros: {trainer._logs("valid")}')

    util.print_box('Evaluate on test')
    trainer.evaluate(
        lambda: topic_lm.valid_input_fn(
            pg.get_paths('test'), batch_size=1))
    loss = trainer._logs('valid')['mean_loss']
    ppl = tf.exp(loss)
    logger.error('loss', loss.numpy(), 'ppl', ppl.numpy())
    trainer._reset_metrics()
    logger.warning(f'Should be zeros: {trainer._logs("valid")}')

    util.print_box('Evaluate on train')
    trainer.evaluate(
        lambda: topic_lm.valid_input_fn(
            pg.get_paths('train'), batch_size=1))
    loss = trainer._logs('valid')['mean_loss']
    ppl = tf.exp(loss)
    logger.error('loss', loss.numpy(), 'ppl', ppl.numpy())
    trainer._reset_metrics()
    logger.warning(f'Should be zeros: {trainer._logs("valid")}')


def main():
    # Create.
    topic_lm = TopicLanguageModel(
        iris_config=config,
        ldamodel_path=montrealtools_dir.parent_dir().join(
            f'{args.source}_ldamodel', f'{args.source}.ldamodel'),
        vocab_path=stuff_dir.get_file_path('vocab.txt'))

    if args.mode == 'train':
        train(topic_lm)
    elif args.mode == 'predict':
        inputs, _ = load_arrays()
        _ = topic_lm(inputs)
        topic_lm.load_weights(args.weights_path)
        interactive_prompt(topic_lm)
    elif args.mode == 'sample':
        inputs, _ = load_arrays()
        _ = topic_lm(inputs)
        topic_lm.load_weights(args.weights_path)
        sampling_config.print('Sampling Config')
        sample = sample_sequence(topic_lm)
        ids_to_str(sample.numpy())
    elif args.mode == 'eval':
        inputs, _ = load_arrays()
        _ = topic_lm(inputs)
        topic_lm.load_weights(args.weights_path)
        evaluate(topic_lm)


    def print_topic_stuff(word):
        probs = topic_lm.topic_probs(word)
        logger.info(f'Prob(z | {word}) = {probs}')
        logger.info(f'sum(Prob(z | {word})) = {sum(probs)}')

    # logger.info(f'LDA model has {topic_lm.ldamodel.num_topics} topics.')
    # print_topic_stuff('cat')


if __name__ == '__main__':
    main()
