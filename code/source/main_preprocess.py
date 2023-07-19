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

from argparse import ArgumentParser

from preprocess import preprocess_embeddings
from dataHandling.sofc_exp_utils import get_data_split_docids
from dataHandling.dataLoader import get_vocab

parser = ArgumentParser()
# data location
parser.add_argument('-corpus_dir', default='../sofc-exp-corpus', type=str)
parser.add_argument('-corpus_meta_data_file', default='../sofc-exp-corpus/SOFC-Exp-Metadata.csv', type=str)

# pretrained embedding location
parser.add_argument('-embedding_file_word2vec', default='../data/embeddings/GoogleNews-vectors-negative300.bin')
parser.add_argument('-embedding_file_mat2vec', default='../data/embeddings/pretrained_embeddings')
parser.add_argument('-embedding_file_bpe', default='../data/embeddings/en.wiki.bpe.vs200000.d300.w2v.bin')
parser.add_argument('-embedding_model_bpe', default='../data/embeddings/en.wiki.bpe.vs200000.model')

# output location
parser.add_argument('-output_word2index_file_mat2vec', default='../data/embeddings/word2index_mat2vec.pickle', type=str)
parser.add_argument('-output_word2index_file_word2vec', default='../data/embeddings/word2index_word2vec.pickle', type=str)
parser.add_argument('-output_word2index_file_bpe', default='../data/embeddings/word2index_bpe.pickle', type=str)
parser.add_argument('-output_embedding_file_word2vec', default='../data/embeddings/word2vec.npy', type=str)
parser.add_argument('-output_embedding_file_mat2vec', default='../data/embeddings/mat2vec.npy', type=str)
parser.add_argument('-output_embedding_file_bpe', default='../data/embeddings/bpe.npy', type=str)

args = parser.parse_args()

print(args)
# convert args to a dictionary
options = vars(args)

mat2vec_file = options['embedding_file_mat2vec']
word2vec_file = options['embedding_file_word2vec']
bpe_file = options['embedding_file_bpe']

train_ids, dev_ids, test_ids = get_data_split_docids(options['corpus_meta_data_file'])

data_vocabulary = get_vocab(options['corpus_dir'], train_ids + dev_ids + test_ids)

mat2vec_word2index, mat2vec_matrix = preprocess_embeddings.get_embedding_weight_matrix(mat2vec_file, "mat2vec", data_vocabulary)
word2vec_word2index, word2vec_matrix = preprocess_embeddings.get_embedding_weight_matrix(word2vec_file, "word2vec", data_vocabulary)
bpe_word2index, bpe_matrix = preprocess_embeddings.get_bpe_embedding_weight_matrix(bpe_file, options["embedding_model_bpe"], data_vocabulary)

preprocess_embeddings.save_word2index(mat2vec_word2index, options['output_word2index_file_mat2vec'])
preprocess_embeddings.save_word2index(word2vec_word2index, options['output_word2index_file_word2vec'])
preprocess_embeddings.save_word2index(bpe_word2index, options['output_word2index_file_bpe'])

preprocess_embeddings.save_weight_matrix(mat2vec_matrix, options['output_embedding_file_mat2vec'])
preprocess_embeddings.save_weight_matrix(word2vec_matrix, options['output_embedding_file_word2vec'])
preprocess_embeddings.save_weight_matrix(bpe_matrix, options['output_embedding_file_bpe'])
