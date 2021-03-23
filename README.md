<!---

    Copyright (c) 2019 Robert Bosch GmbH and its subsidiaries.

-->

# SOFC-Exp Textmining Resources

This repository contains the companion material for the following publication:

> Annemarie Friedrich, Heike Adel, Federico Tomazic, Johannes Hingerl, Renou Benteau,
Anika Maruscyk and Lukas Lange. **The SOFC-Exp Corpus and Neural Approaches to Information Extraction
in the Materials Science Domain.** ACL 2020.

Please cite this paper if using the dataset or the code, and direct any questions regarding the dataset
to [Annemarie Friedrich](mailto:annemarie.friedrich@de.bosch.com), and any questions regarding the code to
[Heike Adel](mailto:heike.adel@de.bosch.com).
The paper can be found at the [ACL Anthology](https://www.aclweb.org/anthology/2020.acl-main.116/) or at
[ArXiv](https://arxiv.org/abs/2006.03039).

## Purpose of this Software 

This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be maintained nor monitored in any way.


## The SOFC-Exp Corpus

The SOFC-Exp corpus contains 45 scientific publications about solid oxide fuel cells (SOFCs),
published between 2013 and 2019 as open-access articles all with a CC-BY license.
The dataset was manually annotated by domain experts with the following information:

* Mentions of relevant experiments have been marked using a graph structure corresponding to instances
of an Experiment frame (similar to the ones used in [FrameNet](https://framenet.icsi.berkeley.edu/fndrupal).)
We assume that an Experiment frame is introduced to the discourse by mentions of words such as _report_, _test_ or
_measure_ (also called the _frame-evoking elements_). The nodes corresponding to the respective tokens are the heads of the graphs representing
the Experiment frame.
* The Experiment frame related to SOFC-Experiments defines a set of 16 possible participant slots.
Participants are annotated as dependents of links between the frame-evoking element and the participant node.
* In addition, we provide coarse-grained entity/concept types for all frame participants, i.e, MATERIAL, VALUE or DEVICE.
Note that this annotation has not been performed on the full texts but only on sentences containing information about relevant experiments,
and a few sentences in addition. In the paper, we run experiments for both tasks only on the set of sentences marked as experiment-describing
in the gold standard, which is admittedly a slightly simplified setting. Entity types are only
partially annotated on other sentences. Slot filling could of course also be evaluated in a fully
automatic setting with automatic experiment sentence detection as a first step.

For further information on the annotation scheme, please refer to our paper and the annotation guidelines.

### Corpus File Formats

Each article is indexed by its PMC ID (see [PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/)).
The `sofc-exp` directory containing the manually annotated corpus is structured as follows:

```
sofc-exp-corpus/
    annotations/
        sentences               Binary information per sentence: Does it describe an experiment?
        entity_types_and_slots  Entity type and slot annotations in BIO format by sentence
                                (for experiment-describing sentences only!)
        frames                  Full frame-style annotations
        tokens                  Character-offset based stand-off annotations for tokenized text
                                (created with StanfordCoreNLP)
    texts/                      Raw texts as extracted from the PDFs, one sentence per line
    docs/
        annotation_guidelines.pdf
        SOFC-Exp-Metadata.csv   Additional metadata / references and links to original documents
    
```

The folders `sentences`, and `entity_types_and_slots` contain information derived from the
original `frame_annotations`, as used in our ACL 2020 paper.

* `sentences` contains one file per article with each line describing one sentence: `sentence_id,
label, begin_char_offset, end_char_offset`. 
The binary label (0 or 1) per line corresponds to whether the sentence describes an SOFC-related
experiment or not. We simply considered all sentences containing at least one frame-evoking element to
be an experiment-describing sentence (label 1).
The character offsets expressing the start end end offsets of each sentence refer to the respective files
in the `texts` directory. __*Note that it is mostly the case that each line in these files contains one
sentence, but there are several cases where sentence annotations include line breaks. Hence, do always use the sentence
annotations given here when working with the other annotation levels!*__
Sentence tokenization was performed by Java's built-in BreakIterator.getSentenceInstance with US locale.


* `tokens` contains stand-off annotations for the tokens in the original texts.
Tokenization was done with Stanford CoreNLP.
The file format is as follows:
`sentence_id, token_id, begin_char_offset, end_char_offset`.
Columns are separated by tabs. Character
offsets always refer to the start of the respective sentence. Counts for `sentence_id` and `token_id` start at 1.


* `entity_types_and_slots` contains the entity type and experiment slot info for the subset of sentences
that describe SOFC-related experiments, i.e., the sentences labeled with 1 above.
This information can also be extracted from the `frames` file, we provide it here for convenience. 
The file format is as follows:
`sentence_id, token_id, begin_char_offset, end_char_offset entity_label slot_label`. 
The columns for `entity_types` and `slots` uses a BIO format. Columns are tab-separated. 

* The full frame annotation is represented as follows in `frames`. The
  frames annotated for each article are represented in a tab-separated
  file. First, the files list all annotated text spans with lines
  prefixed with `SPAN`. The second column corresponds to the span ID,
  the third to its entity/concept type label or `EXPERIMENT` followed by
  colon and a more specific experiment mention type. The fourth column
  refers to the sentence ID (line of sentence in text file, counts start
  at 1), the fifth and sixth column represents the character offsets of
  the begin and end of the span within the sentence. The last column
  adds the text corresponding to the span (for debugging/readability
  purposes).

  ```text
  ...
  SPAN  43  MATERIAL                  19   34   37  SFM
  SPAN  44  EXPERIMENT:previous_work  19   43   52  displayed
  SPAN  45  MATERIAL                  19  120  123  air
  SPAN  46  VALUE                     19  125  136  860 S cm−1
  SPAN  47  MATERIAL                  19  142  150  hydrogen
  SPAN  48  VALUE                     19  152  162  48 S cm−1
  SPAN  49  VALUE                     19  180  191  400600 oC9
  ...
  ```

* After the spans, the file lists each experiment frame instance
Frame instances start with `EXPERIMENT`, followed by an experiment ID in the second column and the span ID
of the corresponding frame-evoking element in the third column.
The following lines, in which the first column is empty, list the slots of the frame.
The second column gives the label of the frame participant (slot) and the last column refers
to the span ID of the dependent/slot.
  ```
  ...
  EXPERIMENT	10	44
	  anode_material	43
	  fuel_used	45
	  conductivity	46
	  fuel_used	47
	  conductivity	48
	  working_temperature	49
  ...
  ```
* After the experiment frame instances, the file lists the additional annotations available with our corpus.
Each of these lines starts with `LINK`, giving the label of the relation in the second column,
and two span IDs referring to the start and end span respectively.
Note that links labeled `same_experiment` and `experiment_variation` are here annotated as links between experiment-evoking
mentions, but conceptually they indicate links between the respective `EXPERIMENTS`.

  ```
  ...
  LINK	thickness	66	65
  ...
  LINK	experiment_variation	98	94
  LINK	same_experiment	100	103
  LINK	coreference	2	3
  LINK	coreference	12	17
  ...
  ```



## Code

### Installation Requirements

We ran our experiments using Python 3.8. You need the following conda packages: `torch`, `numpy` and `scikit-learn`, and the pip package `transformers` (by Huggingface).
See also the exported conda environment (`sofcexp.yml` at the top level of the project).

### Preparing Pretrained Embeddings and Language Models

#### Word2vec, mat2vec, BPE

Download the pretrained [word2vec](https://code.google.com/archive/p/word2vec), [mat2vec](https://github.com/materialsintelligence/mat2vec) and [bpe](https://github.com/bheinzerling/bpemb) embeddings and place them in data/embeddings. If you prefer a different storage location, update the values of the command-line parameters `embedding_file_word2vec`, `embedding_file_mat2vec`, `embedding_file_bpe` in `main_preprocess.py`, accordingly.

word2vec and bpe embeddings are expected in .bin format; for mat2vec embeddings, you will need the whole content of the folder `mat2vec/training/models/pretrained_embeddings` from the mat2vec project.

Run the script `main_preprocess.py`. It will reduce the embeddings to the corpus vocabulary and create word-to-embedding-index files.
The reduced embeddings will be stored as .npy files, the word2index files as pickle files.
The default storage place is again data/embeddings but can be changed via command-line arguments of `main_preprocess.py`.

#### (Sci)BERT

Place the [PyTorch SciBERT model](https://huggingface.co/allenai/scibert_scivocab_uncased) into `models/SciBERT/scibert_scivocab_uncased`.
Make sure this directory contains the files `config.json`, `pytorch_model.bin`, and `vocab.txt`.
If you are using a different BERT model, adapt the value of the parameter `pretrained_bert`, see `main.py`.


### Running Cross Validation Experiments
See scripts in `scripts` folder for configurations for replicating our ACL 2020 experiments.
After your jobs for the individual runs of the CV folds have finished, run the file
```
python source/evaluation/evaluate_cross_validation.y
```
with appropriate command line parameters (see this file for arguments) to collect the predictions from the runs and compute aggregate statistics.

## License

The code in this repository is open-sourced under the AGPL-3.0 license. See the
[LICENSE](code/LICENSE) file for details.
For a list of other open source components included in this project, see the
file [code/3rd-party-licenses.txt](code/3rd-party-licenses.txt).

The manual annotations created for the SOFC-Exp corpus located in the
folder [sofc-exp-corpus/annotations](sofc-exp-corpus/annotations) are
licensed under a [Creative Commons Attribution 4.0 International
License](http://creativecommons.org/licenses/by/4.0/) (CC-BY-4.0).
