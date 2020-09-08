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


import numpy as np


def compute_eval_scores(conf_matrix, labels):
    """
    :param conf_matrix: num_classes x num_classes confusion matrix, first dimension is the gold standard label,
        second dimension is the predicted label
    :param  labels: list of labels, indices correspond to the indices in the confusion matrix
    :return: accuracy, macro p, r, f, dictionaries with p, r, f by class
    """
    print("labels", labels)
    num_classes = len(labels)
    conf = np.array(conf_matrix)

    p = {}  # precision by class
    r = {}  # recall by class
    f1 = {}  # F1 by class

    micro_tp = 0
    micro_total_gold = 0
    micro_total_pred = 0

    for i in range(num_classes):
        cat = labels[i]
        correct = conf[i, i]
        total_gold = sum(conf[i])
        total_pred = sum(conf[:,i])
        if i > 0:  # do not include negative class
            micro_tp += correct
            micro_total_gold += total_gold
            micro_total_pred += total_pred
        # Define for now: if not predicted this class at all, P=1
        if total_pred == 0:
            p[cat] = 1
        else:
            p[cat] = correct/total_pred * 100
        if total_gold == 0:
            r[cat] = 0
        else:
            r[cat] = correct/total_gold * 100
        # Define for now: if P=0 and R=0 then F1 = 0
        if p[cat] == 0 and r[cat] == 0:
            f1[cat] = 0
        else:
            f1[cat] = 2*p[cat]*r[cat]/(p[cat]+r[cat])

    # micro-averages
    if micro_total_pred == 0:
        micro_p = 0
    else:
        micro_p = micro_tp / micro_total_pred * 100
    if micro_total_gold == 0:
        micro_r = 0
    else:
        micro_r = micro_tp / micro_total_gold * 100
    if micro_p + micro_r == 0:
        micro_f1 = 0
    else:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)

    # macro-averages (each class weighted equally)
    macro_p = sum(p.values()) / num_classes
    macro_r = sum(r.values()) / num_classes
    # macro-avg. F1 is the harmonic mean of macro-p and macro-r
    macro_f1 = 2*macro_p*macro_r/(macro_p+macro_r)

    # accuracy
    correct = 0
    total = 0
    for i in range(num_classes):
        correct += conf[i,i]
        total += sum(conf[i])
    accuracy = correct/total*100

    return accuracy, macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, p, r, f1


def print_results_classification(conf_matrix, dataset_name, num_labels):
    """
    :param conf_matrix: confusion matrix with predictions vs. true labels
    :param dataset_name: name of dataset (for printing only)
    :param num_labels: number of labels
    :return:
    """
    results = compute_eval_scores(conf_matrix, [i for i in range(num_labels)])
    acc, macro_p, macro_r, macro_f, micro_p, micro_r, micro_f1, classwise_p, classwise_r, classwise_f = results
    print(dataset_name + ' F1: {:.1f}'.format(micro_f1))
    print("class-wise results:")
    for cl in range(num_labels):
        print("{0: <11}".format(cl) + " {:7.1f}".format(classwise_p[cl]) + " {:7.1f}".format(classwise_r[cl]) + " {:7.1f}".format(classwise_f[cl]))
    print("macro-avg:", "{:7.1f}".format(macro_p), "{:7.1f}".format(macro_r), "{:7.1f}".format(macro_f))
    for row in conf_matrix:
        print("\t", row)
    return classwise_p, classwise_r, classwise_f
