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


import logging

import torch
from torch import nn

from model.encoder import Encoder


log = logging.getLogger("sequence_tagger")


class SequenceTagger(nn.Module):
    def __init__(self, weights_matrix, embedding_options, options, device):

        super(SequenceTagger, self).__init__()

        print("Initializing sequence tagger ...")

        self.encoder = Encoder(weights_matrix, embedding_options, options, device)
        
        self.linear = torch.nn.Linear(options['hidden_size'] * 2, options['num_labels'])

        self.to(device)
        

    def forward(self, tokens, subtokens, bertTensor, bert_subtoken_mask, lengths, token_lengths):
        sentence_tensor, output_lengths = self.encoder(tokens, subtokens, bertTensor, bert_subtoken_mask, lengths, token_lengths)

        features = self.linear(sentence_tensor)

        assert(not torch.isnan(features).any())
        return features

