'''
filter the data from leapofthought. 
1. shuffle all of the unique implicit_rules, and divide them up between train/dev/test splits. 
2. add (implicit_rule, hypothesis) pairs to each split, ensuring that dev/test hypotheses are not in the train split
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

train_path = os.path.join(data_dir, 'LeapOfThought', 'taxonomic_reasonings_training_mix_train.jsonl')
dev_path = os.path.join(data_dir, 'LeapOfThought', 'taxonomic_reasonings_training_mix_dev.jsonl')
test_path = os.path.join(data_dir, 'LeapOfThought', 'taxonomic_reasonings_training_mix_test.jsonl')

train_points = []
dev_points = []
test_points = []

def to_pseudo_language(triplet):
    # function drawn from https://github.com/alontalmor/LeapOfThought/blob/master/LeapOfThought/resources/teachai_kb.py
    subject = triplet['subject']
    predicate = triplet['predicate']
    object = triplet['object']
    template = None
    rules_templates = {
        '/r/IsA': 'A [subject] is a [object].',
        '/r/Antonym': 'A [subject] is not a [object].',
        '/r/DistinctFrom': 'A [subject] is not a [object].',
        '/r/PartOf': 'A [subject] is part of [object].',
        '/r/CapableOf': 'A [subject] is capable of [object].',
        '/r/Desires': 'A [subject] desires [object].',
        '/r/NotDesires': 'A [subject] does not desire [object].',
        'hypernym': 'A [subject] is a [object].',
        'meronym': 'A [subject] has a [object].',
        'part of': 'A [subject] has a [object]. (CN)',
        }
    template = rules_templates[predicate]    
    output = template.replace('[subject]', subject).replace('[object]', object)
    return output

# read points
# filter out the knowledge only points from train, and require there to be implicit knowledge
train_points = [point for point in jsonlines.open(train_path) if point['context'] != '' and 'implicit_rule' in point['metadata']]
dev_points = [point for point in jsonlines.open(dev_path) if point['context'] != '' and 'implicit_rule' in point['metadata']]
test_points = [point for point in jsonlines.open(test_path) if point['context'] != '' and 'implicit_rule' in point['metadata']]

# get all knowledge, hypothesis pairs
all_data_pairs = []
all_knowledge = []
all_hypotheses = []
all_full_points = []
for points in [train_points, dev_points, test_points]:
    for point in points:
        hypothesis = point['phrase']
        rule = point['metadata']['implicit_rule']
        implicit_rule = to_pseudo_language(rule)
        point['metadata']['separate_sentences'] = {'implicit_rule' : implicit_rule}
        # implicit_rule = point['metadata']['separate_sentences']['implicit_rule']
        data_pair = (hypothesis, implicit_rule)
        if data_pair not in all_data_pairs:
            all_data_pairs.append(data_pair)
            all_full_points.append(point)
        if implicit_rule not in all_knowledge:
            all_knowledge.append(implicit_rule)
        if hypothesis not in all_hypotheses:
            all_hypotheses.append(hypothesis)

print("stats before shuffling:")
print("n data: ", len(all_data_pairs))
print("n knowledge: ", len(all_knowledge))
print("n hypotheses: ", len(all_hypotheses))
n_unique = len(set(all_knowledge + all_hypotheses))
print("n unique knowledge+hypotheses ", n_unique)
print("n overlap knowledge+hypotheses ", len(all_knowledge) + len(all_hypotheses) - n_unique)

# we split both the knowledge and the hypotheses between train and eval splits, so that we save some knowledge for eval splits to use
n_train_knowledge =  math.ceil(.6*len(all_knowledge))
n_dev_knowledge =  math.ceil(.1*len(all_knowledge))
n_test_knowledge = math.ceil(.3*len(all_knowledge))
while n_train_knowledge + n_dev_knowledge + n_test_knowledge > len(all_knowledge):
    n_train_knowledge -= 1
train_knowledge = np.random.choice(all_knowledge, size=n_train_knowledge, replace=False)
remaining_knowledge = np.setdiff1d(all_knowledge, train_knowledge)
dev_knowledge = np.random.choice(remaining_knowledge, size=n_dev_knowledge, replace=False)
remaining_knowledge = np.setdiff1d(remaining_knowledge, dev_knowledge)
test_knowledge = np.random.choice(remaining_knowledge, size=n_test_knowledge, replace=False)

# shuffle all points
all_points = train_points + dev_points + test_points
np.random.seed(0)
np.random.shuffle(all_points)

# first make train points
train_points = []
for point in all_points:
    hypothesis = point['phrase']
    rule = point['metadata']['implicit_rule']
    implicit_rule = to_pseudo_language(rule)
    point['metadata']['separate_sentences'] = {'implicit_rule' : implicit_rule}
    # implicit_rule = point['metadata']['separate_sentences']['implicit_rule']
    data_pair = (hypothesis, implicit_rule)
    if implicit_rule in train_knowledge:
        train_points.append(point) 
train_hypotheses = [point['phrase'] for point in train_points]
train_implicit_rules = [point['metadata']['separate_sentences']['implicit_rule'] for point in train_points]
train_sentences = train_hypotheses + train_implicit_rules

# make dev points
dev_points = []
for point in all_points:
    hypothesis = point['phrase']
    rule = point['metadata']['implicit_rule']
    implicit_rule = to_pseudo_language(rule)
    point['metadata']['separate_sentences'] = {'implicit_rule' : implicit_rule}
    # implicit_rule = point['metadata']['separate_sentences']['implicit_rule']
    data_pair = (hypothesis, implicit_rule)
    if implicit_rule in dev_knowledge and hypothesis not in train_sentences and implicit_rule not in train_sentences:
        dev_points.append(point) 

# make test points
test_points = []
for point in all_points:
    hypothesis = point['phrase']
    rule = point['metadata']['implicit_rule']
    implicit_rule = to_pseudo_language(rule)
    point['metadata']['separate_sentences'] = {'implicit_rule' : implicit_rule}
    # implicit_rule = point['metadata']['separate_sentences']['implicit_rule']
    data_pair = (hypothesis, implicit_rule)
    # if ' straw ' in hypothesis or ' straw ' in implicit_rule:
    #     print(hypothesis)
    #     print(implicit_rule)
    #     print(hypothesis in train_sentences)
    #     print(implicit_rule in train_sentences)
    #     import pdb; pdb.set_trace()
    if implicit_rule in test_knowledge and hypothesis not in train_sentences and implicit_rule not in train_sentences:
        test_points.append(point) 

# for train data, if there is any implicit knowledge that does not appear in dev/test data, add it in as its own datapoint
max_added_knowledge = 0
add_knowledge_only_points = []
all_dev_knowledge = [point['metadata']['separate_sentences']['implicit_rule'] for point in dev_points]
all_test_knowledge = [point['metadata']['separate_sentences']['implicit_rule'] for point in test_points]
for point in train_points:
    implicit_rule = point['metadata']['separate_sentences']['implicit_rule']
    if implicit_rule not in all_dev_knowledge + all_test_knowledge and len(add_knowledge_only_points) < max_added_knowledge:
        new_point = {
            'phrase' : implicit_rule,
            'metadata' : {},
            'answer' : 1,
            'context' : '',
        }
        add_knowledge_only_points.append(new_point)
print(f"Adding {len(add_knowledge_only_points)} knowledge-only points to train data")
train_points = train_points + add_knowledge_only_points

print("n total points: ", len(train_points) + len(dev_points) + len(test_points))
print("train size: ", len(train_points))
print("dev size: ", len(dev_points))
print("test size: ", len(test_points))
print(f"train prop true: ", round(np.mean([point["answer"] == 1 for point in train_points]),2) )
print(f"dev prop true: ", round(np.mean([point["answer"] == 1 for point in dev_points]),2) )
print(f"test prop true: ", round(np.mean([point["answer"] == 1 for point in test_points]),2) )

train_path = os.path.join(data_dir, 'LeapOfThought', 'taxonomic_reasonings_training_mix_shuffled_train.jsonl')
dev_path = os.path.join(data_dir, 'LeapOfThought', 'taxonomic_reasonings_training_mix_shuffled_dev.jsonl')
test_path = os.path.join(data_dir, 'LeapOfThought', 'taxonomic_reasonings_training_mix_shuffled_test.jsonl')

with jsonlines.open(train_path, mode='w') as file:
    for point in train_points:
        file.write(point)

with jsonlines.open(dev_path, mode='w') as file:
    for point in dev_points:
        file.write(point)

with jsonlines.open(test_path, mode='w') as file:
    for point in test_points:
        file.write(point)