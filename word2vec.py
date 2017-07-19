###############################################################################
# Training Word2Vec to get word embedding vector
#
# Some code are taken from:
# https://github.com/wendykan/DeepLearningMovies 
###############################################################################

import re
import logging
import sys
import os
from gensim.models import Word2Vec
from bs4 import BeautifulSoup

from settings import *


def strip_tags(html):
    """
    Strip html tags
    """
    return BeautifulSoup(html, 'lxml').get_text(' ')


def text_to_token(text):
    """
    Get list of token for training
    """
    # Strip HTML
    text = strip_tags(text)
    # Keep only word
    text = re.sub("\W", " ", text)
    # Lower and split sentence
    token = text.lower().split()
    # Don't remember the number
    for i in range(len(token)):
        token[i] = len(token[i]) * 'DIGIT' if token[i].isdigit() else token[i]
    return token


def read_sentences(fp):
    """
    Read and split token from text file
    """
    sents = []
    with open(fp, 'rt') as f:
        for line in f:
            if '|' not in line:  # Remove menu items in some newspaper
                sents.append(text_to_token(line.strip()))
    return sents


if __name__ == '__main__':
    file_list = sys.argv[1:]
    sentences = []
    for name in file_list:
        # Read data from files
        print('Reading data...')
        if os.path.isdir(name):
            children = os.listdir(name)
            for c in children:
                p = os.path.join(name, c)
                if os.path.isdir(p):
                    continue
                print(p)
                sentences.extend(read_sentences(p))
            continue
        else:
            sentences.extend(read_sentences(name))

    print('Loaded {} sentences!'.format(len(sentences)))
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    # Set values for various parameters
    num_features = VEC_SIZE  # Word vector dimensionality
    min_word_count = MIN_WORD_THRES  # Minimum word count
    num_workers = WORKERS  # Number of threads to run in parallel
    context = WINDOW_SIZE  # Context window size
    downsampling = DOWN_SAMPLING  # Downsample setting for frequent words
    print("Training Word2Vec model...")
    # Initialize and train the model
    model = Word2Vec(sentences, workers=num_workers,
                     size=num_features, min_count=min_word_count,
                     window=context, sample=downsampling, seed=1)
    model.init_sims(replace=True)
    model_name = "var/features_vlsp.vec"
    model.save(model_name)
