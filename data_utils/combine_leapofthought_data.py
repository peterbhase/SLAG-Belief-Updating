'''
long rather than wide format to LeapOfThought data
a single point becomes two points, with main : implicit_rule and main : hypothesis as two separate main inputs
'''

import os
import jsonlines
import numpy as np
import math
from copy import deepcopy

if os.environ.get('DATA_DIR') is not None:
    data_dir = os.environ.get('DATA_DIR')
else:
    data_dir = 'data/'

train_path = os.path.join(data_dir, 'LeapOfThought', 'taxonomic_reasonings_training_mix_shuffled_train.jsonl')
dev_path = os.path.join(data_dir, 'LeapOfThought', 'taxonomic_reasonings_training_mix_shuffled_dev.jsonl')
test_path = os.path.join(data_dir, 'LeapOfThought', 'taxonomic_reasonings_training_mix_shuffled_test.jsonl')

train_points = [point for point in jsonlines.open(train_path) if point['context'] != '' and 'implicit_rule' in point['metadata']]
dev_points = [point for point in jsonlines.open(dev_path) if point['context'] != '' and 'implicit_rule' in point['metadata']]
test_points = [point for point in jsonlines.open(test_path) if point['context'] != '' and 'implicit_rule' in point['metadata']]

print("train size: ", len(train_points))
print("dev size: ", len(dev_points))
print("test size: ", len(test_points))

new_train_points = []
new_dev_points = []
new_test_points = []

print("combining data...")

for new_points, points in zip([new_train_points, new_dev_points, new_test_points], 
                              [train_points, dev_points, test_points]):
    for point in points:
        for main in ['implicit_rule', 'phrase']:
            new_point = deepcopy(point)
            if main == 'phrase':
                new_point['main'] = point['phrase']
                new_point['answer'] = point['answer']
            elif main == 'implicit_rule':
                new_point['main'] = point['metadata']['separate_sentences']['implicit_rule']
                new_point['answer'] = 1
            new_points.append(new_point)

train_path = os.path.join(data_dir, 'LeapOfThought', 'taxonomic_reasonings_training_mix_combined_train.jsonl')
dev_path = os.path.join(data_dir, 'LeapOfThought', 'taxonomic_reasonings_training_mix_combined_dev.jsonl')
test_path = os.path.join(data_dir, 'LeapOfThought', 'taxonomic_reasonings_training_mix_combined_test.jsonl')

print("n train points: ", len(new_train_points))
print("n dev points: ", len(new_dev_points))
print("n test points: ", len(new_test_points))

with jsonlines.open(train_path, mode='w') as file:
    for point in new_train_points:
        file.write(point)

with jsonlines.open(dev_path, mode='w') as file:
    for point in new_dev_points:
        file.write(point)

with jsonlines.open(test_path, mode='w') as file:
    for point in new_test_points:
        file.write(point)