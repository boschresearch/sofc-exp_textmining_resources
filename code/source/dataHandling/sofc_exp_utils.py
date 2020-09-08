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


from collections import defaultdict
import csv


def modify_cross_val_data_split(doc_info, num_folds, fold):
    """
    :param doc_info: corpus metadata dictionary (as created by get_sofc_corpus_metadata
    :param num_folds: total number of cross validation folds
    :param fold: the fold for which to return the train/dev split, folds are indexed by 1, 2, ...
    :return: the modified doc_info object
    """
    # sort training document IDs alphabetically
    train_ids = sorted([docid for docid in doc_info if doc_info[docid]["datasplit"] in set(["train", "dev"])])
    # create folds
    fold -= 1  # function called with 1 = first fold etc.
    # set all documents to train
    for docid in train_ids:
        doc_info[docid]["datasplit"] = "train"
    # pick the fold's dev documents
    for i in range(fold, len(train_ids), num_folds):
        docid = train_ids[i]
        doc_info[docid]["datasplit"] = "dev"


def get_sofc_corpus_metadata(meta_csv_file):
    """
     Read list of all documents with licensing information, predefined data splits etc.
    :param meta_csv_file: CSV file with data split information.
    :return: a dictionary with the relevant information on data split, licensing, annotator for each document
    """
    doc_info = defaultdict(dict)
    with open(meta_csv_file, encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t')
        header = next(csvreader)
        for row in csvreader:
            docid = row[header.index("name")]
            doc_info[docid]["license"] = row[header.index("license")]
            doc_info[docid]["datasplit"] = row[header.index("set")]
    return doc_info


def get_data_split_docids(meta_csv_file, num_cross_val_folds=None, current_cross_val_fold=None):
    """
    :param meta_csv_file: CSV file with data split information
    :param num_cross_val_folds: if using cross validation, specify total number of splits
    :param fold: the current fold
    :return: returns the train, dev and test ids to use in this experiment.
    """
    # retrieve data split as defined in metadata
    doc_info = get_sofc_corpus_metadata(meta_csv_file)
    if num_cross_val_folds is not None:
        print("document info:", len(doc_info))
        modify_cross_val_data_split(doc_info, num_cross_val_folds, current_cross_val_fold)
    train_ids, dev_ids, test_ids = [], [], []
    for docid in doc_info:
        if doc_info[docid]["datasplit"] == "train":
            train_ids.append(docid)
        elif doc_info[docid]["datasplit"] == "dev":
            dev_ids.append(docid)
        elif doc_info[docid]["datasplit"] == "test":
            test_ids.append(docid)
    return train_ids, dev_ids, test_ids
