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


import sys, os
import json


from argparse import ArgumentParser
from datetime import datetime
import torch
from torch import nn, optim
import numpy as np
from torch.utils.data import DataLoader

from model.bert_models import BertSentenceClassifier
from model.bert_models import BertForTokenClassification
from model.sentenceClassifier import SentenceClassifier
from model.sequenceTagger import SequenceTagger
from model.crfLayer import CRFLayer
from dataHandling.dataLoader import SofcExpDataset
from trainers.sentence_classification_trainer import SentenceClassificationTrainer, BertSentenceClassificationTrainer
from trainers.sequence_tagging_trainer import SequenceTaggingTrainer
from trainers.general_trainer import collate_fn
from dataHandling.sofc_exp_utils import get_data_split_docids
from preprocess import preprocess_embeddings
from transformers import AdamW


import constants

if __name__ == '__main__':

    print("EXPERIMENT STARTING ...")

    print("Torch version:", torch.__version__)
    print("Cuda version: ", torch.version.cuda)

    # get begin timestamp
    now = datetime.now()
    begin_time = now.strftime("%m-%d-%Y-%H-%M-%S")

    parser = ArgumentParser()
    # data location
    parser.add_argument('-corpus_dir', default='../../sofc-exp-corpus', type=str)
    parser.add_argument('-corpus_meta_data_file', default='../../sofc-exp-corpus/SOFC-Exp-Metadata.csv', type=str)

    parser.add_argument('-embedding_file_word2vec', default='../../data/embeddings/word2vec.npy', type=str) 
    parser.add_argument('-embedding_file_mat2vec', default='../../data/embeddings/mat2vec.npy', type=str)
    parser.add_argument('-embedding_file_bpe', default='../../data/embeddings/bpe.npy', type=str)
    parser.add_argument('-word2index_file_word2vec', default='../../data/embeddings/word2index_word2vec.pickle', type=str)
    parser.add_argument('-word2index_file_mat2vec', default='../../data/embeddings/word2index_mat2vec.pickle', type=str)
    parser.add_argument('-word2index_file_bpe', default='../../data/embeddings/word2index_bpe.pickle', type=str)
    parser.add_argument('-save_dir', default='../models', type=str, help="directory for model checkpoints and prediction results")

    # Training parameters
    parser.add_argument('-seed', default=300, type=int)
    parser.add_argument('-batch_size', default=10, type=int)
    parser.add_argument('-use_cuda', action='store_true', default=False)
    parser.add_argument('-lr', default=1e-4, type=float)
    parser.add_argument('-weight_decay', default=1e-2, type=float, help='weight decay rate for optimizer')
    parser.add_argument('-task', default='sentence', type=str,
                        help='"sentence", "entity_typing", "slot_typing"')
    parser.add_argument('-optim', default='adam', help='\"sgd\", \"adam\", \"adamW\"')
    parser.add_argument('-max_grad_norm', default=1.0, type=float, help='max norm for clipping gradients')
    parser.add_argument('-adam_epsilon', default=1e-8, type=float)
    parser.add_argument('-max_sent_length', type=int, default=None,
                        help="if given, this is applied to the original tokens for training and dev set (not to test!)")
    parser.add_argument('-freeze_embeddings', action='store_true', default=False, help='freeze word embeddings')

    parser.add_argument('-embeddings', default='word2vec,mat2vec,bpe,bert',
                        help='which embeddings to use, separate them with comma, possible values: word2vec,mat2vec,bpe,bert')
    parser.add_argument('-attention_size', default=50, type=int, help="size of attention layer")
    parser.add_argument('-epochs', default=40, type=int)
    parser.add_argument('-hidden_size', default=50, type=int, help='size of LSTM hidden states')
    parser.add_argument('-num_layers', default=1, type=int, help='number of LSTM layers')

    # Data set related
    parser.add_argument('-subsampling', type=float, default=0.3,
                        help='percentage by which to undersample 0 samples during training')
    parser.add_argument("-num_cross_val_folds", type=int, default=None,
                        help="specifies total number of cross validation folds")
    parser.add_argument("-current_cross_val_fold", type=int, default=None,
                        help="specifies the current cross validation fold")

    # Select model type, paths to pretrained models
    parser.add_argument("-model_type", type=str, default=None, help="Is required, should be one of BiLSTM, BERT")
    parser.add_argument("-pretrained_bert", type=str, help="Path to pretrained BERT model",
                        default="../../data/models/SciBERT/scibert_scivocab_uncased")
    parser.add_argument("-lr_bert", type=float, default=4e-7, help="Learning rate for BERT part")

    args = parser.parse_args()

    # convert args to a dictionary
    options = vars(args)

    print("##### SETTINGS ######")
    for key in options:
        print("{0: >40}  {1}".format(key, options[key]))
    print("#####################\n")

    # set torch, numpy seeds
    torch.manual_seed(options['seed'])
    np.random.seed(options['seed'])

    if options["use_cuda"]:
        try:
            device = torch.device("cuda")
            torch.cuda.manual_seed(options['seed'])
        except:
            print("Error connecting to CUDA. Defaulting to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print('Using device ', device)

    # Set number of classes
    if options['task'] == 'sentence':
        options['num_labels'] = 2
    elif options['task'] == 'entity_typing':
        options['num_labels'] = 9 + 2  # + 2 because of CRF
    elif options['task'] == 'slot_typing':
        options['num_labels'] = 37 + 2  # + 2 because of CRF
    else:
        raise("task " + options['task'] + " not implemented yet")

    options['embeddings'] = options['embeddings'].split(',')

    # in case of cross validation, modify model saving directory
    # will also write prediction results into this saving directory
    print("save_dir", options["save_dir"])
    print("cross validation? fold", options["current_cross_val_fold"])
    if options["current_cross_val_fold"] is not None:
        options['save_dir'] = os.path.join(options['save_dir'], "cv_" + str(options["current_cross_val_fold"]))
        print("save_dir cross validation:", options["save_dir"])

    # load weights matrices
    weights_matrix_w2v = preprocess_embeddings.load_weight_matrix(options['embedding_file_word2vec'])
    weights_matrix_m2v = preprocess_embeddings.load_weight_matrix(options['embedding_file_mat2vec'])
    weights_matrix_bpe = preprocess_embeddings.load_weight_matrix(options['embedding_file_bpe'])
    print('Completed loading weights')

    embedding_options = []
    embedding_weights = []

    w2v_options = {'name': 'word2vec', 'weights': weights_matrix_w2v, 'embedding_type': 'pretrained', 'emb_index': 0,
                   'num_embeddings': weights_matrix_w2v.shape[0], 'embedding_dim': weights_matrix_w2v.shape[1],
                   'padding_idx': constants.PAD_IDX}

    if "word2vec" in options['embeddings']:
        embedding_options.append(w2v_options)
        embedding_weights.append(weights_matrix_w2v)
    m2v_options = {'name': 'mat2vec', 'weights': weights_matrix_m2v, 'embedding_type': 'pretrained', 'emb_index': 1,
                   'num_embeddings': weights_matrix_m2v.shape[0], 'embedding_dim': weights_matrix_m2v.shape[1],
                   'padding_idx': constants.PAD_IDX}
    if "mat2vec" in options['embeddings']:
        embedding_options.append(m2v_options)
        embedding_weights.append(weights_matrix_m2v)
    bpe_options = {'name': 'bpe', 'weights': weights_matrix_bpe, 'embedding_type': 'pretrained_subword', 'emb_index': 0,
                   'num_embeddings': weights_matrix_bpe.shape[0], 'embedding_dim': weights_matrix_bpe.shape[1],
                   'padding_idx': constants.PAD_IDX}
    if "bpe" in options['embeddings']:
        embedding_options.append(bpe_options)
        embedding_weights.append(weights_matrix_bpe)
    bert_options = {"name" : "bert", "embedding_type" : "bert"}
    if "bert" in options["embeddings"]:
        embedding_options.append(bert_options)

    word2index_word2vec = preprocess_embeddings.load_word2index(options['word2index_file_word2vec'])
    word2index_mat2vec = preprocess_embeddings.load_word2index(options['word2index_file_mat2vec'])
    word2index_bpe = preprocess_embeddings.load_word2index(options['word2index_file_bpe'])
    word2indices = [word2index_word2vec, word2index_mat2vec, word2index_bpe]

    # READ DATASET
    # Read predefined data splits
    train_ids, dev_ids, test_ids = get_data_split_docids(options['corpus_meta_data_file'],
                                                         options["num_cross_val_folds"],
                                                         options["current_cross_val_fold"])
    dataset_train = SofcExpDataset(options["task"], options["corpus_dir"], train_ids, word2indices,
                                   options["max_sent_length"], options["pretrained_bert"])
    if options['task'] == 'sentence':
        print("Size of training set", len(dataset_train.data['tokens']), len(dataset_train.data["experiment_label"]),
            sum(dataset_train.data["experiment_label"]))
    elif options['task'] == 'entity_typing':
        print("Size of training set", len(dataset_train.data['tokens']), len(dataset_train.data["entity_label"]))
    elif options['task'] == 'slot_typing':
        print("Size of training set", len(dataset_train.data['tokens']), len(dataset_train.data["slot_label"]))

    dataset_dev = SofcExpDataset(options["task"], options["corpus_dir"], dev_ids, word2indices,
                                 options["max_sent_length"], options["pretrained_bert"])
    print("Size of dev set", len(dataset_dev.data['tokens']))
    dataset_test = SofcExpDataset(options["task"], options["corpus_dir"], test_ids, word2indices, None,
                                  options["pretrained_bert"])
    print("Size of test set", len(dataset_test.data['tokens']))

    print("Generating BERT features")
    max_length = 0  # this means no cut-off
    if options["task"] == "sentence":
        max_length = 300  # BERT input length
    dataset_train.convert_to_bert_features(max_seq_length=max_length)
    dataset_dev.convert_to_bert_features(max_seq_length=max_length)
    dataset_test.convert_to_bert_features(max_seq_length=max_length)

    # CONFIGURE MODEL, LOSS AND OPTIMIZER
    model_list = []

    if options["task"] == "sentence":
        print("SENTENCE TAGGING TASK")

        if options["model_type"] == "BERT":
            # use BERT tensor as input
            dataset_train = dataset_train.data_pytorch["bertTensor"]
            dataset_dev = dataset_dev.data_pytorch["bertTensor"]
            dataset_test = dataset_test.data_pytorch["bertTensor"]

            model = BertSentenceClassifier.from_pretrained(options["pretrained_bert"])
            model.train()  # set pretrained model to training mode
            model.to(device)
            parameters = model.parameters()
            loss_function = nn.CrossEntropyLoss()
            loss_function.to(device)
            model_list = [model]

        elif options['model_type'] == "BiLSTM":
            model = SentenceClassifier(weights_matrix=embedding_weights, embedding_options=embedding_options,
                                       options=options, device=device)
            model.to(device)  # needs to be done before instantiating the optimizer
            parameters = model.parameters()
            loss_function = nn.CrossEntropyLoss()
            loss_function.to(device)
            model_list = [model]

        else:
            raise ("Invalid model type:", options["model_type"])

    elif options['task'] == "entity_typing" or options['task'] == "slot_typing":

        print("SEQUENCE TAGGING TASK")
        if options["model_type"]  == "BERT":
            print("Bert as first model")
            model = BertForTokenClassification.from_pretrained(options['pretrained_bert'], num_labels=options["num_labels"])
            model.set_options(options)
            model.to(device)
        elif options["model_type"] == "BiLSTM":
            print("LSTM with stacked embeddings")
            model = SequenceTagger(weights_matrix=embedding_weights, embedding_options=embedding_options, options=options, device=device)
            model.to(device)
        else:
            raise("Invalid model type:", options["model_type"])

        print("adding CRF")
        crfLayer = CRFLayer(options, device=device)
        crfLayer.to(device)
        parameters = list(model.parameters()) + list(crfLayer.parameters())
        loss_function = crfLayer._calculate_loss
        model_list = [model, crfLayer]

    if options["model_type"] == "BERT" or 'bert' in options['embeddings']:
        print("Using different learning rate and decay options for BERT parameters.")
        optimizer_grouped_parameters = [
            {'params': [p for m in model_list for n, p in m.named_parameters() if not "bert" in n],
             'weight_decay': options['weight_decay'], 'lr': options['lr']},
            {'params': [p for m in model_list for n, p in m.named_parameters() if "bert" in n], 'weight_decay': 0.0, 'lr': options['lr_bert']}
        ]
        parameters = optimizer_grouped_parameters

    # Create output directory if necessary.
    if not os.path.isdir(options["save_dir"]):
        os.makedirs(options["save_dir"], exist_ok=True)

    # Configure optimizer. AdamW works best when using BERT embeddings.
    if options["optim"] == 'sgd':
        print('using SGD optimizer')
        optimizer = optim.SGD(parameters, lr=options["lr"], weight_decay=options["weight_decay"])
    elif options["optim"] == 'adamW':
        print('using AdamW optimizer')
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer = AdamW(parameters, lr=options['lr'], eps=options['adam_epsilon'])
    elif options["optim"] == 'adam':
        print('using Adam optimizer')
        optimizer = optim.Adam(parameters, lr=options["lr"], weight_decay=options["weight_decay"])

    # Configure training strategy.
    if options['task'] == "sentence":

        subsample = True  # subsampling more frequent class in training data
        if options["model_type"] == "BERT":
            trainer = BertSentenceClassificationTrainer(model_list=model_list, loss_function=loss_function, optimizer=optimizer,
                                                    train_set=dataset_train, dev_set=dataset_dev, options=options)

        elif options['model_type'] == "BiLSTM":
            trainer = SentenceClassificationTrainer(model_list=model_list, loss_function=loss_function, optimizer=optimizer,
                                                train_set=dataset_train, dev_set=dataset_dev, options=options)

    elif options['task'] == "entity_typing" or options['task'] == "slot_typing":
        subsample = False
        trainer = SequenceTaggingTrainer(model_list=model_list, loss_function=loss_function, optimizer=optimizer,
                                  train_set=dataset_train, dev_set=dataset_dev, options=options)

    if "bert" in options["embeddings"]:
        model_dir = options['save_dir']
    else:
        model_dir = options['save_dir']
        with open(options['save_dir'] + "/config.json", 'w') as config_out:
            json.dump(options, config_out)
    print("Directory for saving models: ", model_dir)

    print("**************************")
    print("* STARTING TRAINING ...  *")
    print("**************************")
    trainer.train(save_model_dir=model_dir, device=device, subsample=subsample)


    print("### EVALUATION ... ###")

    if options['task'] == 'entity_typing' or options['task'] == "slot_typing" or options['task'] == "typing_fine_orig":
        print("re-loading best model")
        model_state, crf_state = torch.load(os.path.join(model_dir, "model-weigths.bin"))
        model.load_state_dict(model_state)
        crfLayer.load_state_dict(crf_state)
        trainer.update_model(model_list=[model,crfLayer])
        print("evaluating on dev")
        dev_loader = DataLoader(dataset_dev, batch_size=options["batch_size"], collate_fn=collate_fn)
        trainer.evaluate(data_loader=dev_loader, dataset_name="DEV", device=device)
        print("evaluating on test")
        test_loader = DataLoader(dataset_test, batch_size=1, collate_fn=collate_fn)
        trainer.evaluate(data_loader=test_loader, dataset_name="TEST", device=device)

    elif options["task"] == "sentence" and options["model_type"] == "BERT":
        print("re-loading best model")
        model = BertSentenceClassifier.from_pretrained(model_dir)
        model.to(device)
        trainer.update_model(model_list=[model])
        print("evaluating on dev")
        dev_loader = DataLoader(dataset_dev, batch_size=options["batch_size"])
        trainer.evaluate(data_loader=dev_loader, dataset_name="DEV", device=device)
        print("evaluating on test")
        test_loader = DataLoader(dataset_test, batch_size=1)
        trainer.evaluate(data_loader=test_loader, dataset_name="TEST", device=device)
    elif options["task"] == "sentence":
        print("re-loading best model")
        model_state = torch.load(os.path.join(model_dir, "model-weigths.bin"))
        model.load_state_dict(model_state)
        trainer.update_model(model_list=[model])
        print("evaluating on dev")
        dev_loader = DataLoader(dataset_dev, batch_size=options["batch_size"])
        trainer.evaluate(data_loader=dev_loader, dataset_name="DEV", device=device)
        print("evaluating on test")
        test_loader = DataLoader(dataset_test, batch_size=1)
        trainer.evaluate(data_loader=test_loader, dataset_name="TEST", device=device)

