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
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
from trainers.general_trainer import Trainer
from utils import print_results_classification
import copy
from model.sentenceClassifier import SentenceClassifier

class SentenceClassificationTrainer(Trainer):

    def __init__(self, model_list, loss_function, optimizer, train_set, dev_set, options):
        super(SentenceClassificationTrainer, self).__init__(model_list, loss_function, optimizer, train_set, dev_set, options)

    def update_model(self, model_list):
        self.model_list[0] = copy.deepcopy(model_list[0])

    def apply_model(self, batch, device):
        input_batch = batch['token_indices']
        tokens = input_batch.to(device)
        subtokens = batch['subtoken_indices'].to(device)
        length = batch['length'].to(device)
        token_length = batch['token_length'].to(device)

        bertTensorInput = None  # token IDs for BERT
        if "bert" in self.options["embeddings"]:
            bertInput = batch["bertTensor"]
            bertTensorInput = torch.stack([bertInput[0], bertInput[1], bertInput[2]], dim=0).to(device) # make tensor from list
            bert_subtoken_mask = bertInput[5].to(device)
        else:
            bertInput = None
            bertTensorInput = None
            bert_subtoken_mask = None

        if type(self.model_list[0]) is SentenceClassifier:
            scores = self.model_list[0](tokens, subtokens, bertTensorInput, bert_subtoken_mask, length, token_length).squeeze(dim=1)
        else:
            model = self.model_list[0]
            batch = tuple(t.to(device) for t in batch)

            inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2]}
            scores, hidden = model(**inputs)

        predictions = scores.argmax(1).cpu().numpy()
        return scores, predictions

    def get_gold_labels(self, batch, device):
        labels = batch['experiment_label'].to(device)
        labels_detached = labels.detach().cpu().numpy()
        return labels, labels_detached

    def apply_loss_function(self, scores, labels, batch, device):
        loss = self.loss_function(scores, labels)
        return loss

    def print_results(self, conf, dataset_name):
        p, r, f = print_results_classification(conf, dataset_name, self.options["num_labels"])
        return f[1]

    def save_model(self, save_model_dir):
        torch.save(self.model_list[0].state_dict(), os.path.join(save_model_dir, "model-weigths.bin"))


class BertSentenceClassificationTrainer(SentenceClassificationTrainer):

    """
    print_results, loss function as above
    """

    def __init__(self, model_list, loss_function, optimizer, train_set, dev_set, options):
        super(BertSentenceClassificationTrainer, self).__init__(model_list, loss_function, optimizer, train_set, dev_set, options)

    def get_gold_labels(self, batch, device):
        labels = batch[3].to(device)
        labels_detached = labels.detach().cpu().numpy()
        return labels, labels_detached

    def save_model(self, save_model_file):
        # a directory in this case!
        self.model_list[0].save_pretrained(save_model_file)

    def apply_model(self, batch, device):
        model = self.model_list[0]
        batch = tuple(t.to(device) for t in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2]}
        scores, hidden = model(**inputs)
        predictions = scores.argmax(1).cpu().numpy()
        return scores, predictions
