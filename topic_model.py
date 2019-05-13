#!/usr/bin/env python3.6

import json
import os
import fnmatch

import nltk
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from model_util import io_util, util
import string
import random

import gensim
from gensim import corpora

PROJ = io_util.Directory(os.getenv('PROJ'))

# file_path = PROJ.join('data', 'cnn.com')
file_path = PROJ.join('data', 'foxnews.com')
files_json = fnmatch.filter(os.listdir(file_path), '*.json')

files_json = random.sample(files_json, 100)
num_files = len(files_json)

count = 0
d = {}
for ind in range(0, len(files_json)):
    parts = files_json[ind].split(".")
    d[parts[2]] = 1

num_article_topics = len(d)
print('num_article_topics', num_article_topics)


all_articles = []
num_articles_desired = len(files_json)
for ind in range(0, num_articles_desired):
    path = file_path + '/' + files_json[ind]
    with open(path, 'r') as f:
        datastore = json.load(f)
        article = datastore.get('article')
        all_articles.append(article)

m = len(all_articles)

stop = set(stopwords)
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in all_articles]

remove_words = ['said', 'could', 'would', 'get', 'u', 'also', 'one']
new_doc_clean = []
for doc in doc_clean:
    new_doc = list(doc)
    for item in remove_words:
        if item in doc:
            print(len(doc))
            print(item, "removed")
#             doc.remove(item)
#             print item, "removed"
            new_doc = list(filter(lambda a: a != item, new_doc))
            print(len(new_doc))
    new_doc_clean.append(new_doc)


def main():
    # Creating the term dictionary of our courpus, where every unique
    # term is assigned an index.
    dictionary = corpora.Dictionary(new_doc_clean)

    # Converting list of documents (corpus) into Document Term Matrix
    # using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in new_doc_clean]

    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel

    # Running and Trainign LDA model on the document term matrix.
    ldamodel = Lda(
        doc_term_matrix,
        num_topics=num_article_topics,
        id2word=dictionary,
        passes=50)

    result = ldamodel.print_topics(num_topics=num_article_topics, num_words=10)
    for entry in result:
        print(entry)

    ldamodel.save('fox.ldamodel')


if __name__ == '__main__':
    main()
