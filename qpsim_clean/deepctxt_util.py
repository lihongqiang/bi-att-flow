# -*- coding: utf-8 -*-
from __future__ import absolute_import

import string
import sys
import numpy as np
from six.moves import range
from six.moves import zip
import random
from gensim import models

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    '''Pads each sequence to the same length:
    the length of the longest sequence.

    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.

    Supports post-padding and pre-padding (default).

    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.

    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    '''
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

def load_raw_data_x_y(path="./raw_data.tsv", y_shift=-1):
    X = []
    Y = []
    f = open(path, "r", encoding='utf-8')
    for l in f:
        line = l.replace("\n", "")
        fields = line.split('\t')
        if len(fields) != 2:
            continue
        x = fields[0]
        y = int(fields[1]) + y_shift
        if len(x) <= 0:
            continue
        X.append(x)
        Y.append(y)
    f.close()
    return (X, Y)

def load_raw_data_x1_x2_y(path="./raw_data.tsv", y_shift=0):
    X1 = []
    X2 = []
    Y = []
    f = open(path, "r", encoding='utf-8')
    for l in f:
        line = l.replace("\n", "")
        fields = line.split('\t')
        if len(fields) != 3:
            continue
        x1 = fields[0]
        x2 = fields[1]
        y = int(fields[2]) + y_shift
        if len(x1) <= 0:
            continue
        if len(x2) <= 0:
            continue
        X1.append(x1)
        X2.append(x2)
        Y.append(y)
    f.close()
    return (X1,X2,Y)

def is_in_vocab(vocabs, t):
    terms = t.split(' ')
    for term in terms:
        if not term in vocabs:
            return False
    return True

def load_raw_data_termx(path="./raw_data.tsv", y_shift=0, seed=1337, vocabs=None):
    X = []
    X2 = []
    Y = []
    f = open(path, "r", encoding='utf-8')
    for l in f:
        line = l.replace("\n", "")
        fields = line.split('\t')
        if len(fields) != 4:
            continue
        x = fields[0]
        src = fields[1]
        tgt = fields[2]
        y = int(fields[3]) + y_shift
        if len(x) <= 0:
            continue
        x2 = x.replace(src, tgt)
        if x == x2:
            continue

        if vocabs != None:
            if (not is_in_vocab(vocabs, src)) or (not is_in_vocab(vocabs, tgt)):
                continue

        X.append(x)
        X2.append(x2)
        Y.append(y)

    f.close()

    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(X2)
    np.random.seed(seed)
    np.random.shuffle(Y)

    return (X, X2, Y)

def base_filter():
    f = string.punctuation
    f = f.replace("'", '')
    f += '\t\n'
    return f

def text_to_word_sequence(text, filters=base_filter(), lower=True, split=" "):
    '''prune: sequence of characters to filter out
    '''
    if lower:
        text = text.lower()
    text = text.translate(maketrans(filters, split*len(filters)))
    seq = text.split(split)
    return [_f for _f in seq if _f]

def get_words_uniq(texts):
    vocabs = set()
    for text in texts:
        seq = text_to_word_sequence(text)
        for word in seq:
            if word not in vocabs:
                vocabs.add(word)
    return vocabs 

class DCTokenizer(object):
    def __init__(self, nb_words=None, filters=base_filter(),
                 lower=True, split=' '):

        # reserve inex 0..9 for special purpose
        # 0 -> NOT_USED
        # 1 -> OOV
        # 2 -> BEGIN 
        # 3 -> reserve
        # 4 -> reserve
        # 5 -> reserve
        # 6 -> reserve
        # 7 -> reserve
        # 8 -> reserve
        # 9 -> reserve

        self.vocab_dim = -1
        self.n_symbols = -1
        self.word2index = {}
        self.embedding_weights = None
        self.index_oov = -1
        self.index_begin = -1
        self.nb_words = nb_words
        self.filters = filters
        self.lower = lower
        self.split = split

    def load_bin(self, filename, vocabs=None, vocab_size=40000):
        self.word2index = {}

        self.word2index["_NOT_USED_"] = 0
        self.word2index["_OOV_"] = 1
        self.word2index["_BEGIN_"] = 2
        self.word2index["_RESV3_"] = 3
        self.word2index["_RESV4_"] = 4
        self.word2index["_RESV5_"] = 5
        self.word2index["_RESV6_"] = 6
        self.word2index["_RESV7_"] = 7
        self.word2index["_RESV8_"] = 8
        self.word2index["_RESV9_"] = 9

        self.index_oov = self.word2index["_OOV_"]
        self.index_begin = self.word2index["_BEGIN_"]

        self.vocab_dim = -1

        __vocab_size = vocab_size

        with open(filename, "rb") as f:
            header = f.readline()
            vocab_size, vocab_dim = map(int, header.split())

        self.n_symbols = len(self.word2index) + min(__vocab_size, vocab_size)
        self.vocab_dim = vocab_dim

        #Read in binary word2vec via Gensim
        self.embedding_weights = np.zeros((self.n_symbols, self.vocab_dim))
        w2v_model = models.Word2Vec.load_word2vec_format(filename, binary=True, limit=__vocab_size)
        index = len(self.word2index)
        for word in w2v_model.vocab:
            weights = np.array(w2v_model[word], dtype=float)
            self.word2index[word] = index
            self.embedding_weights[index,:] = weights
            index += 1

        print("n_symbols=" + str(self.n_symbols))
        print("vocab_dim=" + str(self.vocab_dim))

    def load(self, filename, vocabs=None, vocab_size=40000):
        self.word2index = {}

        self.word2index["_NOT_USED_"] = 0
        self.word2index["_OOV_"] = 1
        self.word2index["_BEGIN_"] = 2
        self.word2index["_RESV3_"] = 3
        self.word2index["_RESV4_"] = 4
        self.word2index["_RESV5_"] = 5
        self.word2index["_RESV6_"] = 6
        self.word2index["_RESV7_"] = 7
        self.word2index["_RESV8_"] = 8
        self.word2index["_RESV9_"] = 9

        self.index_oov = self.word2index["_OOV_"]
        self.index_begin = self.word2index["_BEGIN_"]

        self.vocab_dim = -1

        word_count = 0
        with open(filename, 'r', encoding='utf-8') as f_in:
            for l in f_in:
                fields = l.replace("\n","").replace("\t"," ").split(' ')
                word = fields[0]

                if self.vocab_dim < 0:
                    weights = np.fromstring(" ".join(fields[1:]), dtype=float, sep=' ')
                    self.vocab_dim = len(weights)
              
                if vocabs != None and word not in vocabs:
                    continue

                word_count += 1
                if word_count == vocab_size:
                    break
                    
        self.n_symbols = len(self.word2index) + word_count
        self.embedding_weights = np.zeros((self.n_symbols, self.vocab_dim))

        index = len(self.word2index)
        with open(filename, 'r', encoding='utf-8') as f_in:
            for l in f_in:
                fields = l.replace("\n","").replace("\t", " ").split(' ')
                word = fields[0]

                if vocabs != None and word not in vocabs:
                    continue

                weights = np.fromstring(" ".join(fields[1:]), dtype=float, sep=' ')
                self.word2index[word] = index
                self.embedding_weights[index,:] = weights
                index += 1

                if index == vocab_size + 10:
                    break

        print("n_symbols=" + str(self.n_symbols))
        print("vocab_dim=" + str(self.vocab_dim))

    def is_in_vocab(self, t):
        terms = t.split(' ')
        for term in terms:
            if not term in self.word2index:
                return False
        return True

    def texts_to_sequences(self, texts, maxlen):
        '''
            Transform each text in texts in a sequence of integers.
            Only top "nb_words" most frequent words will be taken into account.
            Only words known by the tokenizer will be taken into account.

            Returns a list of sequences.
        '''
        res = []
        for vect in self.texts_to_sequences_generator(texts, maxlen):
            res.append(vect)
        return res

    def texts_to_sequences_generator(self, texts, maxlen):
        '''
            Transform each text in texts in a sequence of integers.
            Only top "nb_words" most frequent words will be taken into account.
            Only words known by the tokenizer will be taken into account.

            Yields individual sequences.
        '''
        nb_words = self.nb_words
        for text in texts:
            seq = text_to_word_sequence(text, self.filters, self.lower, self.split)
            vect = []
            for w in seq:
                i = self.word2index.get(w)
                if i is not None:
                    if nb_words and i >= nb_words:
                        vect.append(self.index_oov)
                    else:
                        vect.append(i)
                else:
                    vect.append(self.index_oov)
                if maxlen > 0 and len(vect) >= maxlen:
                    break
            yield vect

    def texts_to_sequences_with_phrase(self, texts, maxlen):
        '''
            Transform each text in texts in a sequence of integers.
            Only top "nb_words" most frequent words will be taken into account.
            Only words known by the tokenizer will be taken into account.

            Returns a list of sequences.
        '''
        res = []
        for vect in self.texts_to_sequences_generator_with_phrase(texts, maxlen):
            res.append(vect)
        return res

    def texts_to_sequences_generator_with_phrase(self, texts, maxlen):
        '''
            Transform each text in texts in a sequence of integers.
            Only top "nb_words" most frequent words will be taken into account.
            Only words known by the tokenizer will be taken into account.

            Yields individual sequences.
        '''
        nb_words = self.nb_words
        for text in texts:
            seq = text_to_word_sequence(text, self.filters, self.lower, self.split)
            vect = []

            seq_len= len(seq)
            i = 0
            while i < seq_len:

                w = seq[i]
                for j in range(2,5):
                    w_ = '_'.join(seq[i:i+j])
                    if w == w_:
                        break
                    idx = self.word2index.get(w_)
                    if idx is not None:
                        w = w_

                idx = self.word2index.get(w)
                if idx is not None:
                    vect.append(idx)
                else:
                    vect.append(self.index_oov)
                if maxlen > 0 and len(vect) >= maxlen:
                    break

                i += len(w.split('_'))
            yield vect



