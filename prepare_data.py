#!/usr/bin/env python3.6

import argparse
import pickle

from model_util import io_util, util

from model_util.preprocessing import ALLOWED_KINDS
from model_util.preprocessing import path_generator as pg
from model_util.preprocessing import Vectorizer

import stream
from stream import article_generator, VectorizedJSONArticleStream

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_level', default=None, type=str,
    help='Global logging level. "info", "debug", etc.')
parser.add_argument(
    '-b', '--bundle', default=None, type=str)
parser.add_argument(
    '-d', '--data_dir', default='/Users/brandon/Downloads/foxnews.com')
parser.add_argument(
    '-s', '--streamer', default='VectorizedTruncatingJSONArticleStream')
args = parser.parse_args()

root_dir = io_util.Directory.of_file(__file__)

logger = util.get_logger(
    'project',
    is_root_logger=True,
    console_level='debug')

if args.bundle is None:
    args.bundle = root_dir.subdir('models', create=True).join('default')

bundle = io_util.Directory(args.bundle)
if not bundle.exists():
    logger.info(f'Creating bundle: {bundle}')
    bundle.create()


def get_vectorizer(path_generator=None) -> Vectorizer:
    vectorizer = Vectorizer(
        max_num_sents=None,
        max_seq_len=None,
        vocab_size=int(1e6))

    logger.trace('Training vectorizer.')
    vectorizer.update(map(
        str.lower,
        article_generator(path_generator.get_paths('train'))))
    vectorizer.finalize_updates()
    vectorizer.truncate_vocab()

    with open(bundle.join('vocab.txt'), 'w+') as f:
        for word in vectorizer.vocab:
            f.write(word + '\n')

    with open(bundle.join('vectorizer.pkl'), 'wb+') as f:
        pickle.dump(vectorizer, f)

    return vectorizer


class FakeVectorizerForMontrealIDs:

    def __init__(self):
        from model_util.preprocessing import reader
        self.vocab = reader.Vocabulary(
            vocab_file='/Users/brandon/LocalDocuments/models/english_language_model_vocabulary.txt',
            unk_token='UNK')

    def text_to_vector(self, text: str):
        tokens = text.split()
        return [self.vocab.token_to_id(t)
                or self.vocab.token_to_id(self.vocab.unk_token)
                for t in tokens]

    def sequences_to_docs(self, stupid_docs):
        pred_ids = stupid_docs[0][0]
        return [' '.join([self.vocab.id_to_token(i) for i in pred_ids])]

    def sent_tokenize(self, x):
        import nltk
        return nltk.sent_tokenize(x, language='english')


def prepare_data(kind, vectorizer, path_generator):
    streamer = getattr(stream, args.streamer)(vectorizer=vectorizer)

    num_tokens_total = 0
    with open(bundle.join(f'{kind}.txt'), 'w') as f:
        # TODO: stop assuming VectorizedTruncatingJSONArticleStream...
        for sentence_ids in streamer.stream(path_generator.get_paths(kind)):
            num_tokens_total += len(sentence_ids)
            f.write(' '.join(map(str, sentence_ids)) + '\n')
    logger.info(f'Saved {num_tokens_total} {kind} tokens.')


def main():
    data_dir = io_util.Directory(args.data_dir)
    path_generator = pg.FileFlatGenerator(
        data_dir, data_split={
            'train': 0.97,
            'valid': 0.02,
            'test': 0.01})
    for k in ALLOWED_KINDS:
        for path in path_generator.get_paths(k):
            assert path.endswith('.json'), path
    util.print_box('Files per kind', [
        (k, path_generator.num_files(k))
        for k in ALLOWED_KINDS])
    # vectorizer = get_vectorizer(path_generator)
    vectorizer = FakeVectorizerForMontrealIDs()

    for kind in ALLOWED_KINDS:
        prepare_data(kind, vectorizer, path_generator)


if __name__ == '__main__':
    main()

