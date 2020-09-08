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
from trainers.general_trainer import Trainer
from utils import compute_eval_scores
from evaluation import conlleval
import constants
import copy
import numpy as np
import pickle
from model.bert_models import BertForTokenClassification, word_piece_to_input_tokens
from model.sequenceTagger import SequenceTagger


class SequenceTaggingTrainer(Trainer):
    def __init__(self, model_list, loss_function, optimizer, train_set, dev_set, options):
        super(SequenceTaggingTrainer, self).__init__(model_list, loss_function, optimizer, train_set, dev_set, options)

    def update_model(self, model_list):
        self.model_list[0] = copy.deepcopy(model_list[0])
        self.model_list[1] = copy.deepcopy(model_list[1])

    def apply_model(self, batch, device):
        lengths_on_device = batch['length'].to(device)
        input_batch = batch['token_indices']

        # maximum sentence length (real tokens)
        max_sent_length = len(input_batch[0][0])

        bertTensorInput = None  # token IDs for BERT
        if "bert" in self.options["embeddings"]:

            bertInput = batch["bertTensor"]
            bertTensorInput = torch.stack([bertInput[0], bertInput[1], bertInput[2]], dim=0).to(device) # make tensor from list
            bert_subtoken_mask = bertInput[5].to(device)
        else:
            bertInput = None
            bertTensorInput = None
            bert_subtoken_mask = None

        if type(self.model_list[0]) is SequenceTagger:
            scores = self.model_list[0](input_batch.to(device), batch['subtoken_indices'].to(device), bertTensorInput,
                                        bert_subtoken_mask, lengths_on_device, batch['token_length'].to(device)).squeeze(dim=1)
        elif type(self.model_list[0]) is BertForTokenClassification:

            inputs = {'input_ids': bertTensorInput[0],
                      'attention_mask': bertTensorInput[1],
                      'token_type_ids': bertTensorInput[2]
                      }
            subtoken_start_mask = bertInput[5]

            scores, pooled_output = self.model_list[0](**inputs)

            # Change here if we you to use BERT output without classification layer
            # selected_bert_vectors = word_piece_to_input_tokens(pooled_output, subtoken_start_mask, max_sent_length, device)
            selected_bert_logits = word_piece_to_input_tokens(scores, subtoken_start_mask, max_sent_length, device)

            scores = selected_bert_logits # could use BERT vectors here instead

        predictions = self.model_list[1]._obtain_labels(scores, lengths_on_device, device)
        predictions_flat = []
        for batch_pred in predictions:
            predictions_flat.extend(batch_pred)
        return scores, predictions_flat

    def get_gold_labels(self, batch, device):
        if self.options['task'] == 'entity_typing':
            labels = batch['entity_label'].to(device)
        elif self.options['task'] == 'slot_typing':
            labels = batch['slot_label'].to(device)
        labels_flat = []
        for b_idx, batch_labels in enumerate(labels.detach().cpu().numpy()):
            labels_flat.extend(batch_labels[:batch['length'][b_idx].item()])
        return labels, labels_flat

    def apply_loss_function(self, scores, labels, batch, device):
        lengths_on_device = batch['length'].to(device)
        loss = self.loss_function(scores, lengths_on_device, labels, device)
        return loss

    def print_results(self, conf, dataset_name):
        if self.options['task'] == 'entity_typing':
            entity2ind = constants.entity2index
            ind2entity = constants.index2entity
        elif self.options['task'] == 'slot_typing':
            entity2ind = constants.slot2index
            ind2entity = constants.index2slot
        eval_scores = compute_eval_scores(conf, [i for i in range(len(entity2ind.keys()))])
        acc, macro_p, macro_r, macro_f, micro_p, micro_r, micro_f1, classwise_p, classwise_r, classwise_f = eval_scores
        print(dataset_name + ' f1: {:10.2f}'.format(micro_f1))
        print("class-wise results:")
        for cl in sorted(ind2entity.keys()):
            class_name = ind2entity[cl]
            print(class_name, classwise_p[cl], classwise_r[cl], classwise_f[cl])
        print("macro scores:", macro_p, macro_r, macro_f)
        print("conf", conf)
        return micro_f1

    def save_model(self, save_model_dir):
        save_model_file = os.path.join(save_model_dir, "model-weigths.bin")
        torch.save([self.model_list[0].state_dict(), self.model_list[1].state_dict()], save_model_file)

    def evaluate(self, data_loader, dataset_name, device):
        all_labels = np.array([])
        all_predictions = np.array([])
        for model_part in self.model_list:
            model_part.eval()
        for batch in data_loader:
            scores, predictions = self.apply_model(batch, device)
            labels, labels_flat = self.get_gold_labels(batch, device)
            all_labels = np.append(all_labels, labels_flat, axis=0)
            all_predictions = np.append(all_predictions, predictions, axis=0)
        if self.options['task'] == 'entity_typing':
            ind2entity = constants.index2entity
        elif self.options['task'] == 'slot_typing':
            ind2entity = constants.index2slot

        labels_mapped = [ind2entity[all_labels[i]] for i in range(len(all_labels))]
        predictions_mapped = [ind2entity[all_predictions[i]] for i in range(len(all_predictions))]
        
        labels_and_predictions = [" ".join([str(i), labels_mapped[i], predictions_mapped[i]]) for i in range(len(all_labels))]

        labels_and_predictions_old = [" ".join([str(i), ind2entity[all_labels[i]],
                                            ind2entity[all_predictions[i]]]) for i in range(len(all_labels))]

        assert(labels_and_predictions == labels_and_predictions_old)  # sanity check to not introduce error now

        if self.options['task'] == 'slot_typing':
            # remove experiment-evoking verb labels and predictions because we do not want to evaluate it here
            labels_mapped_cleaned = ['O' if labels_mapped[i] in ['B-experiment_evoking_word', 'I-experiment_evoking_word'] else labels_mapped[i] for i in range(len(labels_mapped))]
            predictions_mapped_cleaned = ['O' if predictions_mapped[i] in ['B-experiment_evoking_word', 'I-experiment_evoking_word'] else predictions_mapped[i] for i in range(len(predictions_mapped))]
            labels_and_predictions = [" ".join([str(i), labels_mapped_cleaned[i], predictions_mapped_cleaned[i]]) for i in range(len(all_labels))]

        counts = conlleval.evaluate(labels_and_predictions)
        print("CONLL SCORES")
        print(conlleval.report(counts))
        scores = conlleval.get_scores(counts)
        print(scores)
        # write predictions (of this fold) to file for further evaluation
        pickle.dump((all_labels, all_predictions), open(os.path.join(self.options["save_dir"],
                                                                     "predictions-" + dataset_name + ".p"), 'wb'))
        return conlleval.metrics(counts)[0].fscore

