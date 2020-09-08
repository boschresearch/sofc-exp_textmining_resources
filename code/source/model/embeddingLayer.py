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


import torch
from torch import nn
from model.bert_models import BertForTokenClassification, word_piece_to_input_tokens

class StackedEmbeddings(nn.Module):
    """
    A pytorch layer that supports simple concatenation of a several embedding layer. We currently support:
    - simple pretrained word embeddings, e.g. word2vec
    - word embeddings requiring subword tokenization, e.g. BytePair encodings and BERT
    """

    def __init__(self, embedding_type_list, pretrained_weights, embedding_options, options, device):
        super(StackedEmbeddings, self).__init__()

        self.output_dim = 0
        self.device = device

        supported_embedding_types = set(["character_lstm", "character_cnn", "pretrained_subword", "flair", "elmo",
                                         "pretrained", "bert"])

        module_list = []
        self.embedding_type_list = []
        self.embedding_option_list = []

        for emb_type, pretrained, emb_options in zip(embedding_type_list, pretrained_weights, embedding_options):
            print("adding embedding type:", emb_type)
            if not emb_type in supported_embedding_types:
                print("embedding type", emb_type, "is not supported. Supported types are: ", supported_embedding_types)
                continue
            print("Stacking " + emb_options['name'] + " embeddings")
            if emb_type == "character_lstm":
                char2words = nn.Sequential(
                              nn.Embedding(num_embeddings=emb_options['num_embeddings'],
                                           embedding_dim=emb_options['embedding_dim'],
                                           padding_idx=emb_options['padding_idx']),
                              nn.LSTM()
                           )
                module_list.append(char2words)
                self.output_dim += 2 * options["char_lstm_hiddensize"]
                self.embedding_type_list.append(emb_type)
                self.embedding_option_list.append(emb_options)
            elif emb_type == "character_cnn":
                char2words = nn.Sequential(
                              nn.Embedding(num_embeddings=emb_options['num_embeddings'],
                                           embedding_dim=emb_options['embedding_dim'],
                                           padding_idx=emb_options['padding_idx'])
                              #nn.Conv2D()
                           )
                module_list.append(char2words)
                self.output_dim += options["char_cnn_kernels"]
                self.embedding_type_list.append(emb_type)
                self.embedding_option_list.append(emb_options)
            elif emb_type == "pretrained" or "pretrained_subword":
                emb_layer = nn.Embedding(num_embeddings=emb_options['num_embeddings'],
                                         embedding_dim=emb_options['embedding_dim'],
                                         padding_idx=emb_options['padding_idx'])
                if not pretrained is None:
                    emb_layer.load_state_dict({'weight': pretrained})
                module_list.append(emb_layer)
                self.output_dim += emb_layer.weight.shape[1]
                self.embedding_type_list.append(emb_type)
                self.embedding_option_list.append(emb_options)

        if "bert" in options["embeddings"]:
            print("configure BERT embedding layer")
            emb_layer = BertForTokenClassification.from_pretrained(options["pretrained_bert"])
            module_list.append(emb_layer)
            self.output_dim += emb_layer.hidden_size
            self.embedding_type_list.append("bert")
            self.embedding_option_list.append(None)

        self.embedding_layer = nn.ModuleList(module_list)

    def get_output_dim(self):
        return self.output_dim
            
    def forward(self, token_input_list, subtoken_input_list, bertTensor, bert_subtoken_mask, token_length_list):
        layer_output_list = []
        token_idx = 0
        subtoken_idx = 0
        max_sent_length = len(token_input_list[0][0])
        for layer_idx in range(len(self.embedding_layer)):
            emb_layer = self.embedding_layer[layer_idx]
            layer_type = self.embedding_type_list[layer_idx]

            if layer_type != "bert":
                emb_options = self.embedding_option_list[layer_idx]

            if layer_type == "bert":
                inputs = {'input_ids': bertTensor[0],
                          'attention_mask': bertTensor[1],
                          'token_type_ids': bertTensor[2]
                          }

                logits, pooled_output = emb_layer(**inputs)

                embedded_input = word_piece_to_input_tokens(pooled_output, bert_subtoken_mask, max_sent_length, self.device)

            elif layer_type == "pretrained_subword":
                layer_input = subtoken_input_list.to_dense()
                # layer_input is a sparse tensor but indexing and embedding layers do only allow dense tensors
                layer_input = layer_input[:,subtoken_idx]
                embedded_input = emb_layer(layer_input)
                # merge subword embeddings for each word
                embedded_input = embedded_input.sum(dim=2) / token_length_list[:,subtoken_idx].float().unsqueeze(2)
                assert(not torch.isnan(embedded_input).any())
                subtoken_idx += 1
            else:
                layer_input = token_input_list[:,token_idx]
                embedded_input = emb_layer(layer_input)
                token_idx += 1

            layer_output_list.append(embedded_input)

        if len(layer_output_list) == 1:
            stacked_out = layer_output_list[0]
        else:
            stacked_out = torch.cat(layer_output_list, dim = 2)
        return stacked_out
