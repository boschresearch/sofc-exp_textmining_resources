""" Neural models for information extraction tasks related to the SOFC-Exp corpus (ACL 2020).
Copyright (c) 2020 Robert Bosch GmbH
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


"""
Collect and evaluate cross validation results from parallel runs
"""
import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from argparse import ArgumentParser
import pickle
from sklearn.metrics import confusion_matrix
import numpy as np

import constants
from utils import print_results_classification
from evaluation import conlleval

# Slots actually used in experiments
slot_types = ['anode_material', 'cathode_material', 'conductivity', 'current_density', 'degradation_rate',
                   'device', 'electrolyte_material', 'fuel_used', 'interlayer_material', 'open_circuit_voltage',
                    'power_density', 'resistance', 'support_material', 'thickness', 'time_of_operation',
                    'voltage', 'working_temperature']

entity_types = ["EXPERIMENT", "MATERIAL", "VALUE", "DEVICE"]

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-predictions_dir", default="../data", type=str)  # assumes each sub-folder in this directory contains predictions from one fold
    parser.add_argument(("-eval_mode"), default="multiclass", help="can be one of multiclass (usual P, R, F evaluation) or conll (for BIO sequence tagging)")
    parser.add_argument("-num_labels", default=2, help="number of labels (only applicable for eval_model 'multiclass')")
    parser.add_argument("-task", default="entity_types", help="either entity_types or slot_types (only applicable for eval_mode 'conll')")
    args = parser.parse_args()
    options = vars(args)  # convert args to a dictionary

    pred_dir = options["predictions_dir"]
    eval_mode = options["eval_mode"]
    num_labels = int(options["num_labels"])
    task = options["task"]

    assert eval_mode in ["multiclass", "conll"]
    if eval_mode == "conll":
        assert task in ["entity_types", "slot_types"]

    print("Evaluation mode:", eval_mode)
    print("Predictions directory:", pred_dir)
    print("Found directory?", os.path.isdir(pred_dir))
    print("Task:", task)

    if task == "entity_types":
        label_names = entity_types
    elif task == "slot_types":
        label_names = slot_types

    devData = []
    testData = []
    foldNames = []
    for folder in os.listdir(pred_dir):
        print("Fold:", folder)
        if not os.path.isdir(os.path.join(pred_dir, folder)):
            continue
        foldNames.append(folder)
        devData.append(pickle.load(open(os.path.join(pred_dir, folder, "predictions-DEV.p"), 'rb')))
        testData.append(pickle.load(open(os.path.join(pred_dir, folder, "predictions-TEST.p"), 'rb')))

    cur_modelname = os.path.basename(pred_dir)

    if eval_mode == "multiclass":
        p_dev = []
        r_dev = []
        f1_dev = []
        # for DEV data, we have one prediction for each instance, can use this to compute average P, R, F across folds
        # and look at variance
        for i in range(len(devData)):
            print("\n********* FOLD, dev results **********************")
            labels, predictions = devData[i]
            conf = confusion_matrix(labels, predictions, [i for i in range(num_labels)])
            p_classwise, r_classwise, f1_classwise = print_results_classification(conf, foldNames[i], num_labels)
            p_dev.append(p_classwise[1])  # want to comapre results for "true" class
            r_dev.append(r_classwise[1])
            f1_dev.append(f1_classwise[1])
        mean_p = np.mean(np.array(p_dev))
        std_p = np.std(np.array(p_dev))
        mean_r = np.mean(np.array(r_dev))
        std_r = np.std(np.array(r_dev))
        mean_f1 = np.mean(np.array(f1_dev))
        std_f1 = np.std(np.array(f1_dev))
        print(f1_dev, mean_f1)

        # for TEST data, we have num_cross_val_folds scores, use a simple majority vote to determine final label
        # and test set accuracy
        test_predictions = None
        test_labels = None
        for i in range(len(testData)):
            print("\n********* FOLD, test results **********************")
            labels, predictions = testData[i]
            # Test results from individual runs (just as a sanity check!)
            conf = confusion_matrix(labels, predictions, [i for i in range(num_labels)])
            p_classwise, r_classwise, f1_classwise = print_results_classification(conf, foldNames[i], num_labels)
            p1 = p_classwise[1]
            r1 = r_classwise[1]
            f1 = f1_classwise[1]
            print("Test", p1, r1, f1)
            if test_predictions is None:
                test_predictions = [[] for k in range(len(predictions))]
            if test_labels is None:
                test_labels = labels
            # check if same order
            same_labels = True
            for j in range(len(test_labels)):
                if test_labels[j] != labels[j]:
                    same_labels = False
            assert same_labels
            for j in range(len(labels)):
                test_predictions[j].append(predictions[j])

        # do majority voting
        final_test_predictions = []
        for pred in test_predictions:
            counts = np.bincount(np.array([int(p) for p in pred]))
            label = np.argmax(counts)
            final_test_predictions.append(label)
        conf = confusion_matrix(test_labels, final_test_predictions, [i for i in range(num_labels)])
        p_classwise, r_classwise, f1_classwise = print_results_classification(conf, "TEST-majority-vote", num_labels)
        p1 = p_classwise[1]
        r1 = r_classwise[1]
        f1 = f1_classwise[1]

        print("\n\nResults for dev / test")
        print(cur_modelname + " & {:7.1f}".format(mean_p) + "+/-{:1.1f}".format(std_p) + " & {:7.1f}".format(mean_r) + "+/-{:1.1f}".format(std_r)
              + " & {:7.1f}".format(mean_f1) + "+/-{:1.1f}".format(std_f1)
              + " & {:7.1f}". format(p1) + " & {:7.1f}". format(r1) + " & {:7.1f}". format(f1) + "\\\\")

    elif eval_mode == "conll":
        if task == "slot_types":
            index_to_use = constants.index2slot
        elif task == "entity_types":
            index_to_use = constants.index2entity
        class2p_dev = {}
        class2r_dev = {}
        class2f1_dev = {}
        class2mean_dev = {}
        class2std_dev = {}
        # for DEV data, we have one prediction for each instance, can use this to compute average P, R, F across folds
        # and look at variance
        for i in range(len(devData)):
            print("\n********* FOLD, dev results **********************")
            labels, predictions = devData[i]
            labels_and_predictions = [" ".join([str(i), index_to_use[labels[i]],
                                            index_to_use[predictions[i]]]) for i in range(len(labels))]
            counts = conlleval.evaluate(labels_and_predictions)
            print(conlleval.report(counts))
            scores = conlleval.get_scores(counts)
            for cl in scores:
                if not cl in class2p_dev:
                    class2p_dev[cl] = []
                    class2r_dev[cl] = []
                    class2f1_dev[cl] = []
                class2p_dev[cl].append(scores[cl][0])
                class2r_dev[cl].append(scores[cl][1])
                class2f1_dev[cl].append(scores[cl][2])
        print("**Averaged results for dev**")
        for cl in label_names: # the entity names over which to average
            if not cl in class2p_dev:
                print("\t".join([cl, "{:.1f}".format(0), "+/-{:.1f}".format(0)]))
                class2mean_dev[cl] = 0
                class2std_dev[cl] = 0
                continue
            mean_p = np.mean(np.array(class2p_dev[cl]))
            std_p = np.std(np.array(class2p_dev[cl]))
            mean_r = np.mean(np.array(class2r_dev[cl]))
            std_r = np.std(np.array(class2r_dev[cl]))
            mean_f1 = np.mean(np.array(class2f1_dev[cl]))
            std_f1 = np.std(np.array(class2f1_dev[cl]))
            class2mean_dev[cl] = mean_f1
            class2std_dev[cl] = std_f1
            print("\t".join([cl, "{:.1f}".format(mean_f1), "+/-{:.1f}".format(std_f1)]))

        macro_mean_dev = np.mean(np.array([class2mean_dev[cl] for cl in label_names]))
        macro_std_dev = np.mean(np.array([class2std_dev[cl] for cl in label_names]))
        print("macro avg", macro_mean_dev, macro_std_dev)

        # for TEST data, we also average over folds for now
        class2p_test = {}
        class2r_test = {}
        class2f1_test = {}
        class2mean_test = {}
        for i in range(len(testData)):
            print("\n********* FOLD, test results **********************")
            labels, predictions = testData[i]
            labels_and_predictions = [" ".join([str(i), index_to_use[labels[i]],
                                            index_to_use[predictions[i]]]) for i in range(len(labels))]
            counts = conlleval.evaluate(labels_and_predictions)
            scores = conlleval.get_scores(counts)
            for cl in scores:
                if not cl in class2p_test:
                    class2p_test[cl] = []
                    class2r_test[cl] = []
                    class2f1_test[cl] = []
                class2p_test[cl].append(scores[cl][0])
                class2r_test[cl].append(scores[cl][1])
                class2f1_test[cl].append(scores[cl][2])

        print("Averaged results for test")
        for cl in label_names:
            if not cl in class2p_test:
                print("\t".join([cl, "{:.1f}".format(0), "+/-{:.1f}".format(0)]))
                class2mean_test[cl] = 0
                continue
            mean_p = np.mean(np.array(class2p_test[cl]))
            mean_r = np.mean(np.array(class2r_test[cl]))
            mean_f1 = np.mean(np.array(class2f1_test[cl]))
            std_f1 = np.std(np.array(class2f1_test[cl]))
            class2mean_test[cl] = mean_f1
            print("\t".join([cl, "{:.1f}".format(mean_f1), "+/-{:.1f}".format(std_f1)]))

        macro_mean_test = np.mean(np.array([class2mean_test[cl] for cl in label_names]))
        print("macro avg", macro_mean_test)

        print()
        print("\n\nResults for dev / test")
        output_string = cur_modelname
        for cl in label_names:
            output_string += " & {:7.1f}".format(class2mean_dev[cl]) + "+/-{:1.1f}".format(class2std_dev[cl])
        output_string += " & {:7.1f}".format(macro_mean_dev) + "+/-{:1.1f}".format(macro_std_dev)
        for cl in label_names:
            output_string += " & {:7.1f}".format(class2mean_test[cl])
        output_string += " & {:7.1f}".format(macro_mean_test)
        output_string += "\\\\"
        print(output_string)
