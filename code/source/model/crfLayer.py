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
https://github.com/zalandoresearch/flair/blob/master/flair/models/sequence_tagger_model.py,
licensed under the MIT License
cf. 3rd-party-licenses.txt file in the root directory of this source tree.
"""

import logging

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import nn

from model.embeddingLayer import StackedEmbeddings

import constants

log = logging.getLogger("sequence_tagger")

START_TAG = constants.START_TAG
STOP_TAG = constants.STOP_TAG

def to_scalar(var):
    return var.view(-1).detach().tolist()[0]


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def argmax_batch(vecs):
    _, idx = torch.max(vecs, 1)
    return idx


def log_sum_exp_batch(vecs):
    maxi = torch.max(vecs, 1)[0]
    maxi_bc = maxi[:, None].repeat(1, vecs.shape[1])
    recti_ = torch.log(torch.sum(torch.exp(vecs - maxi_bc), 1))
    return maxi + recti_


def pad_tensors(tensor_list, device):
    ml = max([x.shape[0] for x in tensor_list])
    shape = [len(tensor_list), ml] + list(tensor_list[0].shape[1:])
    template = torch.zeros(*shape, dtype=torch.long, device=device)
    lens_ = [x.shape[0] for x in tensor_list]
    for i, tensor in enumerate(tensor_list):
        template[i, : lens_[i]] = tensor

    return template, lens_


class CRFLayer(nn.Module):
    def __init__(self, options, device):

        super(CRFLayer, self).__init__()

        self.device = device
        self.options = options

        if self.options['task'] == 'entity_typing':
            entity2ind = constants.entity2index
            ind2entity = constants.index2entity
        elif self.options['task'] == 'slot_typing':
            entity2ind = constants.slot2index
            ind2entity = constants.index2slot

        self.num_labels = max(entity2ind.values()) + 1
        next_idx = self.num_labels
        self.label2index = {}
        for e, i in entity2ind.items():
            self.label2index[e] = i
        self.label2index[START_TAG] = next_idx
        next_idx += 1
        self.label2index[STOP_TAG] = next_idx
        self.index2label = {}
        for e, i in ind2entity.items():
            self.index2label[e] = i
        self.index2label[self.label2index[START_TAG]] = START_TAG
        self.index2label[self.label2index[STOP_TAG]] = STOP_TAG

        # set the dictionaries
        self.tag_dictionary = self.label2index
        print(self.tag_dictionary)
        self.tagset_size = self.num_labels + 2

        self.transitions = torch.nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size)
        )
        self.transitions.detach()[
            self.tag_dictionary[START_TAG] :
        ] = -10000
        self.transitions.detach()[
            :, self.tag_dictionary[STOP_TAG]
        ] = -10000


        self.to(device)
        
    def forward(self, input_scores):
        return input_scores
    
    def _score_sentence(self, feats, tags, lens_, device):

        start = torch.LongTensor([self.tag_dictionary[START_TAG]]).to(device)
        start = start[None, :].repeat(tags.shape[0], 1)

        stop = torch.LongTensor([self.tag_dictionary[STOP_TAG]]).to(device)
        stop = stop[None, :].repeat(tags.shape[0], 1)

        pad_start_tags = torch.cat([start, tags], 1)
        pad_stop_tags = torch.cat([tags, stop], 1)

        for i in range(len(lens_)):
            pad_stop_tags[i, lens_[i] :] = self.tag_dictionary[STOP_TAG]

        score = torch.FloatTensor(feats.shape[0]).to(device)

        for i in range(feats.shape[0]):
            r = torch.LongTensor(range(lens_[i])).to(device)

            score[i] = torch.sum(
                self.transitions[
                    pad_stop_tags[i, : lens_[i] + 1], pad_start_tags[i, : lens_[i] + 1]
                ]
            ) + torch.sum(feats[i, r, tags[i, : lens_[i]]])

        return score

    def _calculate_loss(self, scores, lengths, labels, device):
        forward_score = self._forward_alg(scores, lengths, device)
        gold_score = self._score_sentence(scores, labels, lengths, device)
        score = forward_score - gold_score

        return score.mean()


    def _obtain_labels(self, feature, lengths, device):

        tags = []

        for feats, length in zip(feature, lengths):
            confidences, tag_seq = self._viterbi_decode(feats[:length], device)
            tags.append(
                [
                    tag for tag in tag_seq
                ]
            )

        return tags

    def _viterbi_decode(self, feats, device):
        backpointers = []
        backscores = []

        init_vvars = (
            torch.FloatTensor(1, self.tagset_size).to(device).fill_(-10000.0)
        )
        init_vvars[0][self.tag_dictionary[START_TAG]] = 0
        forward_var = init_vvars


        for feat in feats:
            next_tag_var = (
                forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size)
                + self.transitions
            )
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            forward_var = viterbivars_t + feat
            backscores.append(forward_var)
            backpointers.append(bptrs_t)

        terminal_var = (
            forward_var
            + self.transitions[self.tag_dictionary[STOP_TAG]]
        )
        terminal_var.detach()[self.tag_dictionary[STOP_TAG]] = -10000.0
        terminal_var.detach()[
            self.tag_dictionary[START_TAG]
        ] = -10000.0
        best_tag_id = argmax(terminal_var.unsqueeze(0))

        best_path = [best_tag_id]

        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id.item())

        best_scores = []
        for backscore in backscores:
            softmax = F.softmax(backscore, dim=0)
            _, idx = torch.max(backscore, 0)
            prediction = idx.item()
            best_scores.append(softmax[prediction].item())

        start = best_path.pop()
        assert start == self.tag_dictionary[START_TAG]
        best_path.reverse()
        return best_scores, best_path

    def _forward_alg(self, feats, lens_, device):

        init_alphas = torch.FloatTensor(self.tagset_size).fill_(-10000.0)
        init_alphas[self.tag_dictionary[START_TAG]] = 0.0

        forward_var = torch.zeros(
            feats.shape[0],
            feats.shape[1] + 1,
            feats.shape[2],
            dtype=torch.float,
            device=device,
        )

        forward_var[:, 0, :] = init_alphas[None, :].repeat(feats.shape[0], 1)

        transitions = self.transitions.view(
            1, self.transitions.shape[0], self.transitions.shape[1]
        ).repeat(feats.shape[0], 1, 1)

        for i in range(feats.shape[1]):
            emit_score = feats[:, i, :]

            tag_var = (
                emit_score[:, :, None].repeat(1, 1, transitions.shape[2])
                + transitions
                + forward_var[:, i, :][:, :, None]
                .repeat(1, 1, transitions.shape[2])
                .transpose(2, 1)
            )

            max_tag_var, _ = torch.max(tag_var, dim=2)

            tag_var = tag_var - max_tag_var[:, :, None].repeat(
                1, 1, transitions.shape[2]
            )

            agg_ = torch.log(torch.sum(torch.exp(tag_var), dim=2))

            cloned = forward_var.clone()
            cloned[:, i + 1, :] = max_tag_var + agg_

            forward_var = cloned

        forward_var = forward_var[range(forward_var.shape[0]), lens_, :]

        terminal_var = forward_var + self.transitions[
            self.tag_dictionary[STOP_TAG]][None, :].repeat(forward_var.shape[0], 1)

        alpha = log_sum_exp_batch(terminal_var)

        return alpha


    def get_transition_matrix(self):
        data = []
        for to_idx, row in enumerate(self.transitions):
            for from_idx, column in enumerate(row):
                row = [
                    self.index2label[from_idx],
                    self.index2label[to_idx],
                    column.item(),
                ]
                data.append(row)
            data.append(["----"])
        return data
