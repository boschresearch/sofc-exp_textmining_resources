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
from model.attentionLayer import Attention
from model.encoder import Encoder


class SentenceClassifier(nn.Module):
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
        self.num_labels = options['num_labels']
        self.device = device
        self.options = options

        self.encoder = Encoder(weights_matrix, embedding_options, options, device)

        representation_size = 2 * self.hidden_size

        self.attention_size = options['attention_size']
        self.attention = Attention(self.hidden_size*2, self.attention_size)
        self.batchnorm3 = nn.BatchNorm1d(self.hidden_size*2)

        self.linear = nn.Linear(representation_size, self.num_labels)

    def forward(self, tokens, subtokens, bertTensor, bert_subtoken_mask, lengths, token_lengths, return_weights=False):

        outputs, out_lens = self.encoder(tokens, subtokens, bertTensor, bert_subtoken_mask, lengths, token_lengths)

        max_length = outputs.shape[1]

        masks = []
        for length in lengths:
            falses = [False]*length.item()
            trues = [True]*(outputs.shape[1] - length.item())
            mask_tsr = torch.tensor(falses+trues).unsqueeze(0)
            masks.append(mask_tsr)
        masks = torch.cat(masks, dim=0).to(self.device)

        hidden, attention_weights = self.attention(outputs, masks)
        hidden = self.batchnorm3(hidden)

        output = self.linear(hidden)

        if return_weights:
            return output, attention_weights
        else:
            return output
