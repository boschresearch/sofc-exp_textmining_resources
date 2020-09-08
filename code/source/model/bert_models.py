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

from transformers import BertPreTrainedModel, BertModel
from torch import nn
import torch


class BertForTokenClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """
    def __init__(self, config):

        super(BertForTokenClassification, self).__init__(config)
        self.options = {}
        self.max_max_seq_length = 400  # Maximum sequence length in wordpiece tokens that is fed into BERT model at once
        self.stride = 100  # Stride length for computing BERT embeddings
        print("Loading pre-trained BERT model with ", self.config.num_labels, "labels and max seq. length", self.max_max_seq_length)

        self.bert = BertModel(config)
        self.bertDropout = nn.Dropout(config.hidden_dropout_prob)
        self.bertClassifier = nn.Linear(config.hidden_size, config.num_labels)
        self.hidden_size = config.hidden_size
        self.init_weights()

    def set_options(self, options):
        self.options = options
        print("options in bert", options)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):

        # have small BERTs to process sequence in chunks (in case they exceed maximum length of BERT models)
        output_stacked = []
        start = 0
        end = self.max_max_seq_length

        # start + stride = actual start of the sequence we care about
        while start == 0 or start + self.stride < input_ids.shape[1]:

            input_ids_sliced = input_ids[:,start:end]
            attention_mask_sliced = attention_mask[:, start:end]
            token_type_ids_sliced = token_type_ids[:, start:end]

            outputs = self.bert(input_ids_sliced,
                                attention_mask=attention_mask_sliced,
                                token_type_ids=token_type_ids_sliced,
                                position_ids=position_ids,
                                head_mask=head_mask)
            if start == 0:  # no stride in the beginning
                cur_output = outputs[0]
            else:
                cur_output = outputs[0][:,self.stride:,:]
            output_stacked.append(cur_output)
            start = end - self.stride
            end += self.max_max_seq_length

        sequence_output = torch.cat(output_stacked, dim=1)

        sequence_output = self.bertDropout(sequence_output)
        logits = self.bertClassifier(sequence_output)

        return logits, sequence_output


class BertSentenceClassifier(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """
    def __init__(self, config):
        super(BertSentenceClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.bertDropout = nn.Dropout(config.hidden_dropout_prob)
        self.bertClassifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.hidden_size = config.hidden_size
        self.init_weights()

    def get_cls(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        pooled_output = outputs[1]
        return pooled_output

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids)\
            #                head_mask=head_mask)
        pooled_output = outputs[1]
        pooled_output = self.bertDropout(pooled_output)
        logits = self.bertClassifier(pooled_output)  # = a vector with scores for the classes, input for softmax!
        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        return logits, pooled_output


def word_piece_to_input_tokens(bert_output, bert_subtoken_mask, max_sent_length, device):
    """
    :param pooled_output: output vectors from BERT
    :param bert_subtoken_mask: subtoken mask, 1 if BERT token corresponds to *start* of an input token
    :param hidden_size: hidden size of BERT model used
    :param max_sent_length: maximum length of output vectors (sentence length in real tokens)
    :param device: CUDA or CPU
    :return: vectors taken from the pooled output sequence, selecting only the ones corresponding to the start tokens
    (for an entire batch)
    """

    hidden_size = bert_output.shape[-1]  # hidden size of input vectors

    # pooled output is what we want here (BERT ouput before the classification layer)
    subwords_removed = []

    # print(embedded_input.shape)
    # print(subtoken_start_mask.shape)

    # print(embedded_input)
    # print(subtoken_start_mask)

    for i, sent in enumerate(bert_output):
        # mask off non-relevant tokens: this is used for evaluation only!
        # simply concatenating all sentences here

        boolean_mask = bert_subtoken_mask[i].bool().unsqueeze(1).to(device)
        # masked_select is faster than index_select
        selected_vectors = bert_output[i].masked_select(boolean_mask).view(-1, hidden_size)
        # padding to max. sentence length
        missing_to_lengths = torch.zeros([max_sent_length - len(selected_vectors), hidden_size],
                                         dtype=torch.float).to(device)
        selected_vectors = torch.cat([selected_vectors, missing_to_lengths], dim=0)
        subwords_removed.append(selected_vectors.unsqueeze(0))

    embedded_input = torch.cat(subwords_removed, dim=0)

    return embedded_input
