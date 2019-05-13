#!/usr/bin/env python3.6

import os
import argparse
import pickle

from model_util import io_util, util
from operator import itemgetter

from stream import *
from tensorflow.python import keras

# tf.compat.v1.logging.set_verbosity('INFO')

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_level', default=None, type=str,
    help='Global logging level. "info", "debug", etc.')
parser.add_argument(
    '--bundle', default=None, type=str)
parser.add_argument(
    '--seeds', default=['i', 'the'], nargs='+')
parser.add_argument('--mode', default='interactive')
args = parser.parse_args()

# Build up path to custom config file.
root_dir = io_util.Directory.of_file(__file__)
config_path = os.path.join(args.bundle, 'config.yml')

# Merge custom config file with command-line arguments.
config = util.parse_config(args=args, config_path=config_path)

logger = util.get_logger(
    'project',
    is_root_logger=True,
    console_level=config.log_level)

if args.bundle is None:
    args.bundle = root_dir.subdir('models', create=True).join('default')

bundle = io_util.Directory(args.bundle)
if not bundle.exists():
    raise NotADirectoryError(f'Bundle dir not found: {bundle}')


# beam search
def beam_search_decoder(probs, k):
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in probs:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -np.log(row[j] + 1e-12)]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences


def ramble(model: keras.Sequential, vectorizer: Vectorizer, seed='she'):
    current_token = vectorizer.word_to_index[seed]
    current_seq = [current_token]
    model.reset_states()
    model.summary()

    probs_arr = []

    # for t in range(config.max_seq_len):
    for t in range(5):
        # Format inputs.
        inputs = np.zeros((config.find('batch_size'), 1))
        assert inputs.shape[0] == 32, inputs.shape
        inputs[0, 0] = current_token

        # Forward pass.
        probs = model.predict_on_batch(inputs)[0]
        probs = probs.squeeze()
        probs_arr.append(probs)
        assert probs.shape == (
        vectorizer.vocab_size,), f'{probs.shape} != {vectorizer.vocab_size}'

        # Sample prediction.
        current_token = probs.argmax()
        current_seq.append(current_token)

    text_rant = vectorizer.sequences_to_docs([[current_seq]])
    assert len(text_rant) == 1, len(text_rant)

    # Some post-processing prettification.
    text_rant = util.lmap(str.strip, text_rant[0].split('.'))
    util.print_box('Rant', list(text_rant))

    sequences = beam_search_decoder(probs_arr, k=3)
    sequences = [thing[0] for thing in sequences]
    beam_rant = vectorizer.sequences_to_docs([sequences])
    print('beam rant:', beam_rant)


def get_model_path(bundle: io_util.Directory) -> str:
    if bundle.has_file('model.h5'):
        return bundle.get_file_path('model.h5')
    # TODO: generalize
    return bundle.glob('model_*.h5')[-1]


def feed_token(model: keras.Sequential, token: int):
    # Format inputs.
    inputs = np.zeros((config.find('batch_size'), 1))
    inputs[0, 0] = token

    # Forward pass.
    probs = model.predict_on_batch(inputs)[0]
    probs = probs.squeeze()

    return probs

    # # Sample prediction.
    # return probs.argmax()


def interactive_prompt(model: keras.Sequential,
                       vectorizer: Vectorizer):
    def get_input():
        user_input = input('Enter seed word(s) [Ctrl-C to quit]: ')
        user_input = user_input.lower()
        return user_input

    def one_hot(index):
        res = np.zeros(vectorizer.vocab_size, dtype=np.int64)
        res[index] = 1
        return res


    try:
        while True:
            probs_arr = []

            user_input = get_input()
            model.reset_states()
            pred_ids = []
            for token_id in vectorizer.text_to_vector(user_input):
                pred_ids.append(token_id)
                probs = feed_token(model, token_id)
                pred_id = probs.argmax()
                # probs_arr.append(probs)
                probs_arr.append(one_hot(pred_id))
            pred_ids.append(pred_id)

            for t in range(500):
                probs = feed_token(model, pred_ids[-1])
                # probs_arr.append(probs)
                pred_ids.append(probs.argmax())

            pred_words = vectorizer.sequences_to_docs([[pred_ids]])
            assert len(pred_words) == 1, pred_words
            pred_words = pred_words[0]

            pred_words = ' '.join([
                sent.capitalize()
                for sent in vectorizer.sent_tokenize(pred_words)])

            print(pred_words)

            # print('<>~' * 15)
            # sequences = beam_search_decoder(probs_arr, k=2)
            # sequences = [thing[0] for thing in sequences]
            # beam_rant = vectorizer.sequences_to_docs([sequences])
            # print('beam rant:', beam_rant)
    except KeyboardInterrupt:
        print('Ok bye.')


def main():
    util.print_box('Parameters', sorted(config.items(), key=itemgetter(0)))

    logger.trace('Loading vectorizer.')
    with open(bundle.get_file_path('vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)

    config.update_recursively(vocab_size=vectorizer.vocab_size)
    config.graph.embedding.input_dim = config.vocab_size
    config.graph.dense.units = config.vocab_size

    model_path = get_model_path(bundle)
    logger.warning(f'Using model: {model_path}')
    model = keras.models.load_model(model_path)

    if args.mode == 'ramble':
        for seed in args.seeds:
            ramble(model, vectorizer, seed=seed)
    elif args.mode == 'interactive':
        interactive_prompt(model, vectorizer)


if __name__ == '__main__':
    main()
