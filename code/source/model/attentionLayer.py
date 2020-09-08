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

The class in this file is adapated from
https://github.com/yuhaozhang/tacred-relation,
licensed under the Apache License 2.0
cf. 3rd-party-licenses.txt file in the root directory of this source tree.
"""

import torch
from torch import nn
import torch.nn.functional as F

"""attention layer for use in bilstm framework"""

class Attention(nn.Module):
    """
    A position-augmented attention layer where the attention weight is
    a = T' . tanh(Ux)
    where x is the input.
    """

    def __init__(self, input_size, attn_size):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.attn_size = attn_size
        self.ulinear = nn.Linear(input_size, attn_size)
        self.tlinear = nn.Linear(attn_size, 1)
        self.init_weights()

    def init_weights(self):
        self.ulinear.weight.data.normal_(std=0.001)
        self.tlinear.weight.data.zero_() # use zero to give uniform attention at the beginning

    def forward(self, x, x_mask):
        """
        x : batch_size * seq_len * input_size
        x_mask : same dimensions, but bool tensor. contains true if masked, false if not masked
        """
        batch_size, seq_len, _ = x.size()

        x_proj = self.ulinear(x.contiguous().view(-1, self.input_size)).view(
            batch_size, seq_len, self.attn_size)

        scores = self.tlinear(torch.tanh(x_proj).view(-1, self.attn_size)).view(
            batch_size, seq_len)

        # mask padding
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        weights = F.softmax(scores, dim=1)
        # weighted average input vectors
        outputs = weights.unsqueeze(1).bmm(x).squeeze(1)

        return outputs, weights



