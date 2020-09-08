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


PAD_IDX = 0
OOV_IDX = 1
PAD_TOK = "<PAD>"
OOV_TOK = "<OOV>"

START_TAG = "<START>"
STOP_TAG = "<STOP>"

# for coarse-grained concepts
entity2index = {'O': 0, 'B-DEVICE': 1, 'B-EXPERIMENT': 2, 'B-MATERIAL': 3, 'B-VALUE': 4, 'I-DEVICE': 5, 'I-EXPERIMENT': 6, 'I-MATERIAL': 7, 'I-VALUE': 8}
index2entity = dict((v,k) for k,v in entity2index.items())
index2entity[-1] = "-"  # for nicer debugging output

# for fine-grained experiment slots
slot2index = {'O': 0, "B-conductivity": 1, "I-conductivity": 2, "B-current_density": 3, "I-current_density": 4, "B-degradation_rate": 5, "I-degradation_rate": 6,
              "B-device": 7, "I-device": 8, "B-experiment_evoking_word": 9, "I-experiment_evoking_word": 10, "B-fuel_used": 11, "I-fuel_used": 12, "B-open_circuit_voltage": 13,
              "I-open_circuit_voltage": 14, "B-power_density": 15, "I-power_density": 16, "B-resistance": 17, "I-resistance": 18, "B-thickness": 19, "I-thickness": 20, "B-time_of_operation": 21,
              "I-time_of_operation": 22, "B-voltage": 23, "I-voltage": 24, "B-working_temperature": 25, "I-working_temperature": 26, "B-anode_material": 27, "B-cathode_material": 28,
              "B-electrolyte_material": 29, "B-interlayer_material": 30, "I-anode_material": 31, "I-cathode_material": 32, "I-electrolyte_material": 33,
              "I-interlayer_material": 34, "B-support_material": 35, "I-support_material": 36, "none": 0, "SAME_EXPERIMENT": 0, "B-interconnect_material": 0, "I-interconnect_material": 0}
index2slot = dict((v,k) for k,v in slot2index.items())
index2slot[0] = "O"

RANDOM_SEED = 300
