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
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
from dataHandling.sampling import WeightedDownSampler
import torch
import pickle
from evaluation import conlleval
import constants

def collate_fn(data):
        output = dict()
        output["token_indices"] = torch.stack([t["token_indices"] for t in data])
        output["subtoken_indices"] = torch.stack([t["subtoken_indices"] for t in data])
        output["length"] = torch.stack([t["length"] for t in data])
        output["token_length"] = torch.stack([t["token_length"] for t in data])
        output["tokens"] = [(tokens) for tokens in zip(*[t["tokens"] for t in data])]
        bert_tensors = [t["bertTensor"] for t in data]
        output["bertTensor"] = [torch.stack(bt) for bt in zip(*bert_tensors)]
        output["doc_id"] = [t["doc_id"] for t in data]
        output["sent_id"] = torch.tensor([t["sent_id"] for t in data])
        output["entity_label"] = torch.stack([t["entity_label"] for t in data])
        output["slot_label"] = torch.stack([t["slot_label"] for t in data])
        return output

class Trainer:
    def __init__(self, model_list, loss_function, optimizer, train_set, dev_set, options):
        self.model_list = model_list
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.train_set = train_set
        self.dev_set = dev_set
        self.options = options

    def load_model(self, model_list):
        pass

    def update_model(self, model_list):
        pass  # implement in concrete trainer!

    def apply_model(self, batch, device):
        pass  # implement in concrete trainer!

    def apply_model_structured(self, batch, device):
        pass  # implement in concrete trainer!

    def get_gold_labels(self, batch, device):
        pass  # implement in concrete trainer!

    def apply_loss_function(self, scores, labels, batch, device):
        pass  # implement in concrete trainer!

    def save_model(self, save_model_dir):
        pass  # implement in concrete trainer!

    def print_results(self, conf, dataset_name):
        pass  # implement in concrete trainer!

    def evaluate(self, data_loader, dataset_name, device):
        epoch_labels = np.array([])
        epoch_predictions = np.array([])
        for model_part in self.model_list:
            model_part.eval()
        for batch in data_loader:
            scores, predictions = self.apply_model(batch, device)
            labels, labels_flat = self.get_gold_labels(batch, device)
            epoch_labels = np.append(epoch_labels, labels_flat, axis=0)
            epoch_predictions = np.append(epoch_predictions, predictions, axis=0)
        conf = confusion_matrix(epoch_labels, epoch_predictions, labels=[i for i in range(self.options["num_labels"])])
        micro_f1 = self.print_results(conf, dataset_name)
        # write predictions (of this fold) to file for further evaluation
        pickle.dump((epoch_labels, epoch_predictions), open(os.path.join(self.options["save_dir"],
                                                                         "predictions-" + dataset_name + ".p"), 'wb'))
        return micro_f1

    def get_predictions(self, data_loader, dataset_name, device):
        all_predictions = {}
        for model_part in self.model_list:
            model_part.eval()
        for batch in data_loader:
            doc_id = batch['doc_id']
            sent_id = batch['sent_id']
            scores, predictions = self.apply_model_structured(batch, device)
            for b_idx in range(len(predictions)):
                cur_d = doc_id[b_idx]
                cur_s = sent_id[b_idx].item()
                if not cur_d in all_predictions:
                    all_predictions[cur_d] = {}
                all_predictions[cur_d][cur_s] = predictions[b_idx]
        return all_predictions

    def evaluate_conll(self, data_loader, dataset_name, device):
        epoch_labels = np.array([])
        epoch_predictions = np.array([])
        for model_part in self.model_list:
            model_part.eval()
        for batch in data_loader:
            scores, predictions = self.apply_model(batch, device)
            labels, labels_flat = self.get_gold_labels(batch, device)
            epoch_labels = np.append(epoch_labels, labels_flat, axis=0)
            epoch_predictions = np.append(epoch_predictions, predictions, axis=0)
        conf = confusion_matrix(epoch_labels, epoch_predictions, [i for i in range(self.options["num_labels"])])
        micro_f1 = self.print_results(conf, dataset_name)
        print("CoNLL evaluation:")
        input_conll = ["X\t" + constants.index2entity[int(l)] + "\t" + constants.index2entity[int(p)]
                       for l, p in zip(epoch_labels, epoch_predictions)]
        # write predictions (of this fold) to file for further evaluation
        pickle.dump(input_conll,
                    open(os.path.join(self.options["save_dir"], "predictions-" + dataset_name + ".p"), 'wb'))
        result = conlleval.evaluate(input_conll)
        scores = conlleval.get_scores(result)
        macro_f1 = 0
        for cl in scores:
            macro_f1 += scores[cl][-1]
        macro_f1 = macro_f1 / len(scores)
        print(scores)
        print(macro_f1)
        return macro_f1

    def train(self, save_model_dir, device, subsample=False, subsampleSampler=False, max_grad_norm=1.0):
        dev_loader = None
        train_loader = None
        if self.options["task"] == "entity_typing" or self.options["task"] == "slot_typing":
            dev_loader = DataLoader(self.dev_set, batch_size=self.options["batch_size"], collate_fn=collate_fn)
        else:
            dev_loader = DataLoader(self.dev_set, batch_size=self.options["batch_size"])
        best_f1 = 0.0

        if subsampleSampler:
            rate = self.options["subsampling"]

        print("trainable parameters:")
        model_parameters = []
        for model_part in self.model_list:
            model_parameters += list(model_part.named_parameters())
        for name, param in model_parameters:
            if not param.is_cuda:
                print("parameter", name, "not on gpu")
            if param.requires_grad:
                print(name)

        if subsample:
            # initialize the sampler once, will generate different splits in the training loop.
            rate = self.options["subsampling"]
            if self.options["model_type"] == "BERT":
                # batch data is simply a tensor, need to give class index
                train_sampler = WeightedDownSampler(self.train_set, 3, {1: 0.0, 0: rate})
            else:
                # batch data is a dictionary, need to give key to labels tensor
                train_sampler = WeightedDownSampler(self.train_set, None, {1: 0.0, 0: rate},
                                                    class_key='experiment_label')

        for epoch_counter in range(self.options["epochs"]):
            print('####################################################')
            print('               Training')
            print('                Epoch', epoch_counter)
            print('####################################################')
            if subsample:
                if self.options["task"] == "entity_typing" or self.options["task"] == "slot_typing":
                    train_loader = DataLoader(self.train_set, sampler=train_sampler, batch_size=self.options["batch_size"], collate_fn=collate_fn)
                else:
                    train_loader = DataLoader(self.train_set, sampler=train_sampler, batch_size=self.options["batch_size"])
            else:
                if self.options["task"] == "entity_typing" or self.options["task"] == "slot_typing":
                    train_loader = DataLoader(self.train_set, shuffle=True, batch_size=self.options["batch_size"], collate_fn=collate_fn)
                else:
                    train_loader = DataLoader(self.train_set, shuffle=True, batch_size=self.options["batch_size"])

            # initialize variables to hold epoch loss, labels, scores
            epoch_loss = 0

            for model_part in self.model_list:
                model_part.train()
            batch_counter = 0

            for batch_idx, batch in enumerate(train_loader):
                divisor = 10
                if batch_counter % divisor == 0:
                    print('        Batch', batch_counter, "/", len(train_loader))
                self.optimizer.zero_grad()

                if isinstance(batch, dict): # in case of BERT, it's a TensorDataset
                    if len(batch['token_indices']) == 1:
                        continue  # avoid single-item batches

                scores, _ = self.apply_model(batch, device)
                labels, _ = self.get_gold_labels(batch, device)
                loss = self.apply_loss_function(scores, labels, batch, device)
                epoch_loss += loss.item()
                loss.backward()
                for model_part in self.model_list:
                    torch.nn.utils.clip_grad_norm_(model_part.parameters(), max_grad_norm)
                self.optimizer.step()
                batch_counter += 1
            # evaluation on train
            self.evaluate(train_loader, "TRAIN", device)
            micro_f1 = self.evaluate(dev_loader, "DEV", device)

            # if we've gotten better f1, save weights
            if micro_f1 >= best_f1: # save first iteration in any case
                self.save_model(save_model_dir)
                best_f1 = micro_f1

        print("**** TRAINING DONE ******")
        print("Best F1 during training:", best_f1) # for sentence classification task, this is score of 1 class
        # which score is actually returned depends on the trainer class
