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

from torch.utils.data import Sampler
import random
from copy import deepcopy
from collections import defaultdict


class WeightedDownSampler(Sampler):

    def __init__(self, dataset, class_idx, class_weights, class_key=None):
        """
        :param data_source: The data set to be sampled from.
        :param class_weights: dictionary with downsampling weights for the classes, 0.0 means "keep all", 0.3 means
        "keep 70% of this class"
        :param class_key: if give, use this instead of class index (not tested yet!!) --> for pytorch_all data structures

        Instantiate only once (when creating the Dataset instance).
        In DataLoader, call only the iterator: when instantiating the iterator, the sampling really happens.
        """
        # collect information about where instances of each class are in dataset
        self.class_weights = class_weights
        self.indices_by_class = defaultdict(list)
        for i, inst in enumerate(dataset):
            if class_key:
                label = int(inst[class_key].item())
            else:
                # assume a class_idx is given in this case
                label = int(inst[class_idx].item()) # assume integer coding for classes
            self.indices_by_class[label].append(i)
        # determine length of samples according to the given downsampling weights
        self.num_samples = int(sum([(1-class_weights[c])*len(self.indices_by_class[c]) for c in class_weights]))

    def __iter__(self):
        """
        :return: an iterator over the indices of the items, which will be used sequentially by DataLoader
        to split the data into batches.
        """
        # Now actually downsample the data
        indices = []
        for c in self.indices_by_class:
            class_indices = deepcopy(self.indices_by_class[c])
            random.shuffle(class_indices)
            indices += class_indices[:int(len(class_indices)*(1-self.class_weights[c]))]
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

