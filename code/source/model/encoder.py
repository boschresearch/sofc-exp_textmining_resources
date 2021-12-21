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

import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model.embeddingLayer import StackedEmbeddings


class Encoder(nn.Module):
    """
    bilstm model for experiment sentence classification
    """

    def __init__(self, weights_matrix, embedding_options, options, device):
        """

        :param weights_matrix: numpy matrix with word embeddings
        :param options: python dictionary with hyperparameters
        """

        super().__init__()

        self.hidden_size = options['hidden_size']
        self.device = device

        embedding_types = [emb_opt['embedding_type'] for emb_opt in embedding_options]
        pretrained_embeddings = [torch.from_numpy(emb_opt['weights']).float().to(self.device) if emb_opt['weights'] is not None else None for emb_opt in embedding_options if emb_opt["name"] != "bert"]
        self.embeddingLayer = StackedEmbeddings(embedding_types, pretrained_embeddings, embedding_options, options, device)

        embeddingLayer_out_dim = self.embeddingLayer.get_output_dim()

        self.lstm = nn.LSTM(embeddingLayer_out_dim, 
                                self.hidden_size, 
                                num_layers=options['num_layers'],
                                batch_first=True, 
                                bidirectional=True)


    def forward(self, tokens, subtokens, bertTensor, bert_subtoken_mask, lengths, token_lengths):
        x = self.embeddingLayer(tokens, subtokens, bertTensor, bert_subtoken_mask, token_lengths)

        lengths = lengths.cpu()
        inputs = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        outputs, (ht, ct) = self.lstm(inputs)
        outputs, output_lens = pad_packed_sequence(outputs, batch_first=True)

        return outputs, output_lens
