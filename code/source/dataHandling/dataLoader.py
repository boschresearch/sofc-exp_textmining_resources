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

import os
import csv
import random
import copy
import numpy
import torch
from torch.utils.data import Dataset, TensorDataset
import constants
from transformers import BertTokenizer


def get_vocab(data_path, target_doc_ids):
        """
        Reads in data from corpus files to create a vocabulary for the data.
        Fills a set with word types from the data.
        :param data_path: Path to sofc-exp-corpus with subfolders texts, annotations/experiment_sentence,
            annotations/tokenized, annotations/entity_types and annotations/slots
        :param target_doc_ids: Set of document IDs to be used in this subset of documents (e.g., train, dev or test)
        """
        vocab = set()

        for filename in os.listdir(os.path.join(data_path, "texts")):
            if not filename.endswith(".txt"):
                continue
            else:
                document_id = filename.replace(".txt", "")

            if not document_id in target_doc_ids:
                continue

            # Read text
            with open(os.path.join(data_path, "texts", filename)) as f_text:
                text = f_text.read()

            # Read info per sentence: which sentences are labeled as containing an experiment?
            # Read all sentences of the corpus.
            doc_sentences = {}  # id --> text
            with open(os.path.join(data_path, "annotations", "sentences", document_id + ".csv")) as f_expSent:
                csv_exp_sent = csv.reader(f_expSent, delimiter="\t")
                for row in csv_exp_sent:
                    sent_id = int(row[0])  # sentence ID within document, not unique across dataset
                    label = int(row[1])
                    assert label == 1 or label == 0
                    sent_text = text[int(row[2]):int(row[3])].strip()
                    assert len(sent_text) > 0  # skipping empty sentences if any? - none found in SOFC-Exp dataset
                    doc_sentences[int(row[0])] = sent_text

            # Read tokens for all sentences of the corpus.
            with open(os.path.join(data_path, "annotations", "tokens", document_id + ".csv")) as f_tokenized:
                csv_tokenized = csv.reader(f_tokenized, delimiter="\t", quoting=csv.QUOTE_NONE)
                for row in csv_tokenized:
                    sent_id, token_id, begin, end = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                    token_text = doc_sentences[sent_id][begin:end]
                    vocab.add(token_text)
        return vocab


""" The following class is from 
https://github.com/ThilinaRajapakse/pytorch-transformers-classification.
licensed under the Apache License 2.0
cf. 3rd-party-licenses.txt file in the root directory of this source tree. """
class InputFeatures(object):
    """A single set of features of one instance (for BERT)."""

    def __init__(self, input_ids, input_mask, segment_ids, sent_label_id, sequence_tagging_labels, subtoken_start_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.sent_label_id = sent_label_id
        self.sequence_tagging_labels = sequence_tagging_labels
        self.subtoken_start_mask = subtoken_start_mask


class SofcExpDataset(Dataset):
    """
    Class representing the data of the SOFC-Exp corpus. Reads either al sentences for the sentence classification
    task (experiment sentence identification) or only part of the corpus annotated with entity or slot label
    information (i.e., the sentences labeled as describing an experiment).
    Loads the data from files and generates tensors representing the data (with configured word embeddings or BERT).
    """

    def __init__(self, task, data_path, doc_ids, word2index_list, max_sent_length, bert_tokenizer_path=None):
        random.seed(12345)
        self.max_sent_length = max_sent_length
        self.entity2index = constants.entity2index
        self.word2index_list = word2index_list
        # fill field self.data from corpus text and annotation files
        if task == "sentence":
            self.read_data_sentence_classification(data_path, doc_ids)
        elif task == "entity_typing" or task == "slot_typing":
            self.read_data_sequence_tagging(data_path, doc_ids)
        else:
            raise Exception("'task' must be configured as either sentence, entty_typing or slot_typing.")
        self.prepare_token_indices()  # preparing token indices for embeddings
        self.data = self.padData(self.data)
        self.data['tokens'] = numpy.array(self.data['tokens'])  # in case of lists, dimensions get reversed somehow
        self.data_pytorch = self.convertToTensors(self.data)
        if bert_tokenizer_path:
            self.bertTokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)
        self.convert_to_bert_features() # call this only after the other data loading steps have been performed
        print("Done reading data")

    def __len__(self):
        return len(self.data_pytorch['token_indices'])

    def __getitem__(self, index):
        data = {'token_indices': self.data_pytorch['token_indices'][index],
                'subtoken_indices': self.data_pytorch['subtoken_indices'][index],
                'length': self.data_pytorch['length'][index],
                'token_length': self.data_pytorch['token_length'][index],
                'tokens': self.data['tokens'][index],
                'bertTensor': self.data_pytorch['bertTensor'][index],
                'doc_id': self.data['doc_id'][index],
                'sent_id': self.data['sent_id'][index]}
        if "experiment_label" in self.data_pytorch:
            data["experiment_label"] = self.data_pytorch['experiment_label'][index]
        if "entity_label" in self.data_pytorch:
            data["entity_label"] = self.data_pytorch["entity_label"][index]
        if "slot_label" in self.data_pytorch:
            data["slot_label"] = self.data_pytorch["slot_label"][index]
        return data

    def convertToTensors(self, data):
        data_pytorch = {}
        data_pytorch['token_indices'] = torch.LongTensor(numpy.array(data['token_indices']))
        data_pytorch['subtoken_indices'] = []
        for idx in range(len(data['subtoken_indices'])):
            data_pytorch['subtoken_indices'].append(torch.sparse.LongTensor(data['subtoken_indices'][idx]['i'].t(),
                            data['subtoken_indices'][idx]['v'], torch.Size(data['subtoken_indices'][idx]['size'])))
        data_pytorch['token_length'] = torch.LongTensor(numpy.array(data['token_length']))
        data_pytorch['length'] = torch.LongTensor(numpy.array(data['length']))
        if "experiment_label" in data:
            data_pytorch['experiment_label'] = torch.LongTensor(numpy.array(data['experiment_label']))
        if "entity_label" in data:
            data_pytorch['entity_label'] = torch.LongTensor(numpy.array(data['entity_label']))
        if "slot_label" in data:
            data_pytorch['slot_label'] = torch.LongTensor(numpy.array(data['slot_label']))
        print(data_pytorch.keys())
        return data_pytorch

    def padIndexSequence(self, sequences, length, padding=constants.PAD_IDX, dim=1):
        padded_sequences = copy.deepcopy(sequences)
        if dim == 1:
            for seq in padded_sequences:
                pad_len = length - len(seq)
                if pad_len > 0:
                    seq.extend([padding] * pad_len)
                seq = seq[:length]
        elif dim == 2:
            for dim1 in padded_sequences:
                for seq in dim1:
                    pad_len = length - len(seq)
                    if pad_len > 0:
                        seq.extend([padding] * pad_len)
                    seq = seq[:length]
        elif dim == 3:
            for dim1 in padded_sequences:
                for dim2 in dim1:
                    for seq in dim2:
                        pad_len = length - len(seq)
                        if pad_len > 0:
                            seq.extend([padding] * pad_len)
                        seq = seq[:length]
        return padded_sequences

    def get_sparse_indices_values(self, input_data, max_sentence_length, max_word_length):
        # in the end, we want a list of dictionaries; in particular: one dictionary per data instance
        # the dictionary has three keys:
        # 'i': the indices at which the tensor is non-0
        # 'v': the values at these indices
        # 'size': the size of the densified version of the sparse tensor
        sparse_indices_values = []
        for data_instance in input_data:
            # dim 0: subword embedding
            # dim 1: sentence length
            # dim 2: word length
            cur_dictionary = {}
            cur_indices = []
            cur_values = []
            for subword_embedding_idx in range(len(data_instance)):
                for word_idx in range(len(data_instance[subword_embedding_idx])):
                    for subword_idx in range(len(data_instance[subword_embedding_idx][word_idx])):
                        cur_indices.append([subword_embedding_idx, word_idx, subword_idx])
                        cur_values.append(data_instance[subword_embedding_idx][word_idx][subword_idx])
            cur_dictionary['i'] = torch.LongTensor(cur_indices)
            cur_dictionary['v'] = torch.LongTensor(cur_values)
            cur_dictionary['size'] = (len(data_instance), max_sentence_length, max_word_length)
            sparse_indices_values.append(cur_dictionary)
        return sparse_indices_values

    def padData(self, data):
        max_length = max(data['length'])
        self.max_sent_length = max_length
        print("maximum sentence length", max_length)
        data['token_indices'] = self.padIndexSequence(data['token_indices'], max_length, dim=2)
        lengths = set()
        for b in data['token_indices']:
            lengths.add(len(b))
        max_length_token = 0
        for item in data['token_length']:
            for item_emb_type in item:
                max_length_token = max([max_length_token] + item_emb_type)
        print("maximum token length in subtokens", max_length_token)
        data['subtoken_indices'] = self.get_sparse_indices_values(data['subtoken_indices'], max_length, max_length_token)
        data['token_length'] = self.padIndexSequence(data['token_length'], max_length, padding=1, dim=2)
        if "entity_label" in data:
            data['entity_label'] = self.padIndexSequence(data['entity_label'], max_length)
        if "slot_label" in data:
            data['slot_label'] = self.padIndexSequence(data['slot_label'], max_length)
        return data

    """ The following function is adapted from 
    https://github.com/ThilinaRajapakse/pytorch-transformers-classification.
    licensed under the Apache License 2.0
    cf. 3rd-party-licenses.txt file in the root directory of this source tree. """
    def convert_to_bert_features(self, max_seq_length=0):
        """
        Creates word piece tokenization by applying BERT tokenizer on existing tokens.
        Based on this, creates input tensors for BERT-based models.
        """
        if max_seq_length == 0:
            # compute corresponding max sequence length for BERT
            for i, tokens in enumerate(self.data['tokens']):
                text = " ".join(tokens)
                bert_tokens = self.bertTokenizer.tokenize(text)
                if len(bert_tokens) > max_seq_length:
                    max_seq_length = len(bert_tokens)
            print("Max sequence length for BERT:", max_seq_length)
        # else will cut off sentences at specified max seq.length

        # "config" for BERT
        sep_token = "[SEP]"
        cls_token = "[CLS]"
        sequence_segment_id = 0 # first and only sequence
        pad_token_segment_id = 0

        features = []  # collect InputFeatures

        for i, predefTokens in enumerate(self.data['tokens']):

            tokens = []
            is_original_token_start = []
            for token in predefTokens:
                bertTokens = self.bertTokenizer.tokenize(token)
                if len(bertTokens) == 0:
                    print("no BERT token for:", token)
                    bertTokens = ['"']  # This might be an encoding problem.
                tokens += bertTokens
                is_original_token_start += [1] + [0] * (len(bertTokens)-1)

            # Account for [CLS] and [SEP] with "- 2"
            special_tokens_count = 2
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[:(max_seq_length - special_tokens_count)]
            if len(is_original_token_start) > max_seq_length - special_tokens_count:
                is_original_token_start = is_original_token_start[:(max_seq_length - special_tokens_count)]

            # The convention in BERT is for single sentences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = tokens + [sep_token]
            is_original_token_start = is_original_token_start + [0]
            segment_ids = [sequence_segment_id] * len(tokens)

            tokens = [cls_token] + tokens
            is_original_token_start = [0] + is_original_token_start
            segment_ids = [0] + segment_ids

            input_ids = self.bertTokenizer.convert_tokens_to_ids(tokens)  # here, tokens are converted to numbers

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to. Here: mask padding with zero.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            input_ids = input_ids + [0 for _ in range(padding_length)]
            input_mask = input_mask + [0 for _ in range(padding_length)]
            segment_ids = segment_ids + [0 for _ in range(padding_length)]
            is_original_token_start = is_original_token_start + [0 for _ in range(padding_length)]
            is_original_token_start + [0 for _ in range(padding_length)]

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(is_original_token_start) == max_seq_length

            # collect this instance
            label = None
            sequence_tagging_label_list = None
            if "experiment_label" in self.data_pytorch and len(self.data_pytorch[
                                                                   'experiment_label']) > 0:
                label = self.data_pytorch['experiment_label'][i].item()
            if "entity_label" in self.data_pytorch:
                sequence_tagging_label_list = self.data_pytorch["entity_label"][i].tolist()
            elif "slot_label" in self.data_pytorch:
                sequence_tagging_label_list = self.data_pytorch["slot_label"][i].tolist()
            features.append(InputFeatures(input_ids=input_ids,
                                 input_mask=input_mask,
                                 segment_ids=segment_ids,
                                 sent_label_id=label,
                                 sequence_tagging_labels = sequence_tagging_label_list,
                                 subtoken_start_mask = is_original_token_start
                                          ))

        # After all instances have been tokenized etc., convert to TensorDataSet (input for BERT)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        if "experiment_label" in self.data_pytorch and len(self.data_pytorch['experiment_label']) > 0:
            all_sent_label_ids = torch.tensor([f.sent_label_id for f in features], dtype=torch.long)
        else:
            all_sent_label_ids = torch.tensor([0 for f in features], dtype=torch.long)
        if "entity_label" in self.data_pytorch or "slot_label" in self.data_pytorch:
            all_seq_tagging_label_ids = torch.tensor([f.sequence_tagging_labels for f in features], dtype=torch.long)
        else:
            all_seq_tagging_label_ids = torch.tensor([0 for f in features], dtype=torch.long)
        all_subtoken_start_mask = torch.tensor([f.subtoken_start_mask for f in features], dtype=torch.long)
        bert_tensor_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_sent_label_ids,
                                            all_seq_tagging_label_ids, all_subtoken_start_mask)
        print("Converted to BERT tensor with number sentences/sequences", len(all_input_ids))
        self.data_pytorch["bertTensor"] = bert_tensor_dataset

    def read_data_sentence_classification(self, data_path, target_doc_ids):
        """
        Reads in data from corpus files into a format that is already a preparation for creating tensors.
        Fills a dictionary with keys = data type, and values = lists where each list entry corresponds to one sentence.
        :param data_path: Path to sofc-exp-corpus with subfolders texts, annotations/experiment_sentence,
            annotations/tokenized, annotations/entity_types and annotations/slots
        :param target_doc_ids: Set of document IDs to be used in this subset of documents (e.g., train, dev or test)
        """
        self.data = {'sent_id': [],  # list of sentence IDs
                     'doc_id': [],  # corresponding document ID of sentence
                     'tokens': [],  # tokens of the sentence (text)
                     'token_indices': [],  # *indices of tokens
                     'subtoken_indices': [],  # *subtoken indices (for bpe)
                     'length': [],  # *number of tokens in sentence ?
                     'token_length': [],  # *? lengths of tokens ?
                     'experiment_label': [],  # binary label indicating whether sentence describes an experiment or not
                     'bert_inputs': []}  # *sentence in BERT input format

        for filename in os.listdir(os.path.join(data_path, "texts")):
            if not filename.endswith(".txt"):
                continue
            else:
                document_id = filename.replace(".txt", "")

            if not document_id in target_doc_ids:
                continue

            # Read text
            with open(os.path.join(data_path, "texts", filename)) as f_text:
                text = f_text.read()

            # Read info per sentence: which sentences are labeled as containing an experiment?
            # Read all sentences of the corpus.
            doc_sentences = {}  # id --> text
            with open(os.path.join(data_path, "annotations", "sentences", document_id + ".csv")) as f_expSent:
                csv_exp_sent = csv.reader(f_expSent, delimiter="\t")
                for row in csv_exp_sent:
                    sent_id = int(row[0])  # sentence ID within document, not unique across dataset
                    label = int(row[1])
                    assert label == 1 or label == 0
                    sent_text = text[int(row[2]):int(row[3])].strip()
                    assert len(sent_text) > 0  # skipping empty sentences if any? - none found in SOFC-Exp dataset
                    self.data["experiment_label"].append(label)
                    self.data["doc_id"].append(document_id)
                    self.data["sent_id"].append(sent_id)
                    doc_sentences[int(row[0])] = sent_text

            # Read tokens for all sentences of the corpus.
            with open(os.path.join(data_path, "annotations", "tokens", document_id + ".csv")) as f_tokenized:
                csv_tokenized = csv.reader(f_tokenized, delimiter="\t", quoting=csv.QUOTE_NONE)
                last_sent_id = None
                tokens = []
                for row in csv_tokenized:
                    sent_id, token_id, begin, end = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                    if sent_id != last_sent_id:
                        if not last_sent_id is None:
                            assert len(tokens) > 0
                            self.data["tokens"].append(tokens)
                            self.data["length"].append(len(tokens))
                            tokens = []
                    token_text = doc_sentences[sent_id][begin:end]
                    tokens.append(token_text)
                    last_sent_id = sent_id
                # append info for last sentence
                assert len(tokens) > 0
                self.data["tokens"].append(tokens)
                self.data["length"].append(len(tokens))

    def read_data_sequence_tagging(self, data_path, target_doc_ids):
        """
        Reads in data from corpus files into a format that is already a preparation for creating tensors.
        Fills a dictionary with keys = data type, and values = lists where each list entry corresponds to one sentence.
        This method only reads sentences labeled as containing an Experiment.
        :param data_path: Path to sofc-exp-corpus with subfolders texts, annotations/experiment_sentence,
            annotations/tokenized, annotations/entity_types and annotations/slots
        :param target_doc_ids: Set of document IDs to be used in this subset of documents (e.g., train, dev or test)
        """

        self.data = {'sent_id': [],         # list of sentence IDs
                    'doc_id': [],              # corresponding document ID of sentence
                    'tokens': [],              # tokens of the sentence (text)
                    'token_indices': [],       # *indices of tokens
                    'subtoken_indices': [],    # *subtoken indices (for bpe)
                    'entity_label': [],        # entity labels for tokens in BIO format
                    'entity_labels_text': [],
                    'slot_label': [],          # experiment slot labels in BIO format
                    'slot_labels_text': [],
                    'length': [],              # *number of tokens in sentence ?
                    'token_length': [],        # *? lengths of tokens ?
                    'bert_inputs': []}         # *sentence in BERT input format

        for filename in os.listdir(os.path.join(data_path, "texts")):
            if not filename.endswith(".txt"):
                continue
            else:
                document_id = filename.replace(".txt", "")

            if not document_id in target_doc_ids:
                continue

            # Read text
            with open(os.path.join(data_path, "texts", filename)) as f_text:
                text = f_text.read()

            # Read in pre-segmented sentences
            doc_sentences = {}  # id --> text
            with open(os.path.join(data_path, "annotations", "sentences", document_id + ".csv")) as f_expSent:
                csv_exp_sent = csv.reader(f_expSent, delimiter="\t")
                for row in csv_exp_sent:
                    sent_text = text[int(row[2]):int(row[3])].strip()
                    assert len(sent_text) > 0  # skipping empty sentences if any? - none found in SOFC-Exp dataset
                    doc_sentences[int(row[0])] = sent_text

            # Read tokens, entity labels, slot labels per sentence
            with open(os.path.join(data_path, "annotations", "entity_types_and_slots", document_id + ".csv")) as f_tokenized:
                csv_tokenized = csv.reader(f_tokenized, delimiter="\t", quoting=csv.QUOTE_NONE)
                last_sent_id = None
                tokens = []
                entity_labels = []
                slot_labels = []
                for row in csv_tokenized:
                    sent_id, token_id, begin, end, entity_label, slot_label \
                        = int(row[0]), int(row[1]), int(row[2]), int(row[3]), row[4], row[5]
                    if sent_id != last_sent_id:
                        if not last_sent_id is None:
                            assert len(tokens) > 0
                            self.data["doc_id"].append(document_id)
                            self.data["sent_id"].append(sent_id)
                            self.data["tokens"].append(tokens)
                            self.data["entity_labels_text"].append(entity_labels)
                            self.data["entity_label"].append([constants.entity2index[e] for e in entity_labels])
                            self.data["slot_labels_text"].append(slot_labels)
                            self.data["slot_label"].append([constants.slot2index[s] for s in slot_labels])
                            self.data["length"].append(len(tokens))
                            tokens = []
                            entity_labels = []
                            slot_labels = []
                    token_text = doc_sentences[sent_id][begin:end]
                    tokens.append(token_text)
                    entity_labels.append(entity_label)
                    slot_labels.append(slot_label)
                    last_sent_id = sent_id
                # append info for last sentence
                assert len(tokens) > 0
                self.data["doc_id"].append(document_id)
                self.data["sent_id"].append(sent_id)
                self.data["tokens"].append(tokens)
                self.data["entity_labels_text"].append(entity_labels)
                self.data["entity_label"].append([constants.entity2index[e] for e in entity_labels])
                self.data["slot_labels_text"].append(slot_labels)
                self.data["slot_label"].append([constants.slot2index[s] for s in slot_labels])
                self.data["length"].append(len(tokens))

    def prepare_token_indices(self):
        """
        Prepares the data object further by adding lists corresponding to token indices for the various embeddings.
        """

        # Prepare token indices for different types of embeddings
        token_indices_all = []
        subtoken_indices_all = []
        token_length_all = []
        for sentence in self.data['tokens']:
            sentence_indices = []
            sentence_indices_sub = []
            token_length = []
            for idx, w2i in enumerate(self.word2index_list):
                emb_indices = []
                emb_indices_sub = []
                token_length_sub = []
                for token in sentence:
                    if token in w2i:
                        token_idx = w2i[token]
                    else:
                        if idx == 2:
                            token_idx = [constants.OOV_IDX]
                        else:
                            token_idx = constants.OOV_IDX
                    if isinstance(token_idx, list):
                        emb_indices_sub.append(token_idx)
                        token_length_sub.append(len(token))
                    else:
                        emb_indices.append(token_idx)
                if len(emb_indices_sub) > 0:
                    sentence_indices_sub.append(emb_indices_sub)
                if len(emb_indices) > 0:
                    sentence_indices.append(emb_indices)
                if len(token_length_sub) > 0:
                    token_length.append(token_length_sub)
            token_indices_all.append(sentence_indices)
            subtoken_indices_all.append(sentence_indices_sub)
            token_length_all.append(token_length)
        self.data['token_indices'] = token_indices_all
        self.data['subtoken_indices'] = subtoken_indices_all
        self.data['token_length'] = token_length_all
