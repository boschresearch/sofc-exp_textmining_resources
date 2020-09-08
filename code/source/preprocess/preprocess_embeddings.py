""" Neural models for information extraction tasks related to the SOFC-Exp corpus (ACL 2020).
Copyright (c) 2020 Robert Bosch GmbH
@author: Heike Adel
@author: Annemarie Friedrich

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import pickle
from constants import PAD_TOK, PAD_IDX, OOV_TOK, OOV_IDX
from sentencepiece import SentencePieceProcessor

""" The following function is from
https://github.com/bheinzerling/bpemb/blob/master/bpemb/util.py
licensed under the MIT License
cf. 3rd-party-licenses.txt file in the root directory of this source tree. """
def sentencepiece_load(file):  
    """Load a SentencePiece model"""
    spm = SentencePieceProcessor()
    spm.Load(str(file))
    return spm

def load_weight_matrix(embedding_file):
    """
    :param embedding_file: filename of preprocessed embeddings
    :return: numpy matrix, rows correspond to embeddings of vocab item
    """
    weights_matrix = np.load(embedding_file)

    return weights_matrix

def save_weight_matrix(weight_matrix, embedding_file):
    np.save(embedding_file, weight_matrix)

def get_bpe_embedding_weight_matrix(input_embedding_file, embeddings, data_vocabulary, min_word_freq=1):
    print('creating bpe embedding weights')
    spm = sentencepiece_load("/home/adh2rng/data/embeddings/en.wiki.bpe.vs200000.model")
    wv = KeyedVectors.load_word2vec_format(input_embedding_file, binary=True).wv
    embedding_vocab = wv.vocab

    example_word = wv.index2entity[0]
    embedding_dim = len(wv.get_vector(example_word))
    
    subword2index = {}
    subword2index[PAD_TOK] = PAD_IDX
    subword2index[OOV_TOK] = OOV_IDX
    index = 2

    word2index = {}
    word2index[PAD_TOK] = [PAD_IDX]
    word2index[OOV_TOK] = [OOV_IDX]

    for word in data_vocabulary:
        encoded = spm.EncodeAsPieces(word)
        for subword in encoded:
            if not subword in subword2index:
                subword2index[subword] = index
                index += 1
        word2index[word] = [subword2index[subword] for subword in encoded]
    
    # initialize weights matrix
    matrix_len = len(subword2index)  # assumption: word2index contains PAD_TOK at position 0 and OOV_TOK at position 1
    weights_matrix = np.zeros((matrix_len, embedding_dim))

    # keep track of words in corpus vocab that are not in embeddings
    num_words_found = 0
    words_not_found = []

    for subword in subword2index:
        if subword != OOV_TOK and subword != PAD_TOK:
            try:
                weights_matrix[[subword2index[subword]]] = wv.get_vector(subword)
                num_words_found += 1
            except KeyError:
                words_not_found.append(subword)

    # begin creating new embeddings for those words that were not in m2v or w2v vocab
    mean_embedding = weights_matrix.sum(axis=0) / num_words_found

    weights_matrix[OOV_IDX] = mean_embedding

    return word2index, weights_matrix

def get_embedding_weight_matrix(input_embedding_file, embeddings, data_vocabulary, min_word_freq=1):
    """
    create weight matrix representing word embeddings to be used in BiLSTM classifier
    :param input_embedding_file: file to original word embeddings
    :param embeddings: which embeddings to lead: mat2vec or word2vec
    :param min_word_freq: decides which word2index to use to create embeddings
    """
    if embeddings == 'mat2vec':
        print('creating mat2vec embedding weights')
        wv = Word2Vec.load(input_embedding_file).wv

    else:
        print('creating word2vec embedding weights')
        wv = KeyedVectors.load_word2vec_format(input_embedding_file, binary=True).wv

    example_word = wv.index2entity[0]
    embedding_vocab = wv.vocab
    embedding_dim = len(wv.get_vector(example_word))
    print("embedding dimensionality: ", embedding_dim)

    assert (PAD_TOK not in wv and OOV_TOK not in wv)

    # get word2index
    word2index = {}
    word2index[PAD_TOK] = PAD_IDX
    word2index[OOV_TOK] = OOV_IDX
    index = 2
    for word in data_vocabulary:
        if word in embedding_vocab:
            word2index[word] = index
            index += 1

    # initialize weights matrix
    matrix_len = len(word2index)  # assumption: word2index contains PAD_TOK at position 0 and OOV_TOK at position 1
    weights_matrix = np.zeros((matrix_len, embedding_dim))

    # keep track of words in corpus vocab that are not in embeddings
    num_words_found = 0
    words_not_found = []

    for word in word2index:
        if word != OOV_TOK and word != PAD_TOK:
            try:
                weights_matrix[[word2index[word]]] = wv.get_vector(word)
                num_words_found += 1
            except KeyError:
                words_not_found.append(word)

    # begin creating new embeddings for those words that were not in m2v or w2v vocab
    mean_embedding = weights_matrix.sum(axis=0) / num_words_found

    weights_matrix[OOV_IDX] = mean_embedding

    return word2index, weights_matrix

def load_word2index(word2index_file):
    with open(word2index_file, 'rb') as f:
        word2index = pickle.load(f)
    return word2index

def save_word2index(word2index, word2index_file):
    with open(word2index_file, 'wb') as f:
        pickle.dump(word2index, f)

