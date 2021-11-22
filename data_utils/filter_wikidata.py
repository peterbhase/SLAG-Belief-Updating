'''
filter wikidata5m to triples that use only one of a pre-specified set of relations
'''
import os
import numpy as np
import jsonlines
import sys
import time

allowed_relations = [
    'place of birth',
    'award received',
    'cause of death', 
    'place of death',
    'place of burial',
    'educated at',
    'child',
    'occupation',
    'spouse',
    'sibling',
]

if os.environ.get('DATA_DIR') is not None:
    data_dir = os.environ.get('DATA_DIR')
else:
    data_dir = 'data/'

def filter_for_rare_relations(relations, relation_prop_dict, relation_prop_threshold = .2):
    # filter list of relations to only include those that do not exceed a given threshold for frequency
    return_list = []
    for rel in relations:
        if relation_prop_dict[rel] < relation_prop_threshold:
            return_list.append(rel)
    return return_list

splits = ['train', 'valid', 'test']
load_paths = [os.path.join(data_dir, 'Wikidata5m', f"wikidata5m_transductive_{split}.txt") for split in splits]
save_paths = [os.path.join(data_dir, 'Wikidata5m', f"filtered_wikidata5m_transductive_{split}.jsonl") for split in ['train', 'dev', 'test']]

# make entity and relation dictionaries
start = time.time()
print("Reading data...")
entity_dict = {}
relation_dict = {}
entity_path = os.path.join(data_dir, 'Wikidata5m', 'wikidata5m_entity.txt')
relation_path = os.path.join(data_dir, 'Wikidata5m', 'wikidata5m_relation.txt')
with open(entity_path, 'r') as file:
    for line in file:
        id = line.split()[0]
        entities = [text.strip('\n') for text in line.split('\t')[1:]]
        entity_dict[id] = entities
        entity_info_dict = {}
with open(relation_path, 'r') as file:
    for line in file:
        id = line.split()[0]
        relations = [text.strip('\n') for text in line.split('\t')[1:]]
        relation_dict[id] = relations

seen_rels = []
all_data = []
for load_path in load_paths:
    with open(load_path, 'r') as file:
        for num, line in enumerate(file):
            e1, rel, e2 = [text.strip('\n') for text in line.split('\t')]
            try:
                e1_str = entity_dict[e1][0]
                e2_str = entity_dict[e2][0]
                rel_str = relation_dict[rel][0]
            except:
                # there are many missing entities / relations. they are missing from the entity and relation files
                # if e1 not in entity_dict:  print(f"missing {e1} from entity dictionary")
                # if e2 not in entity_dict:  print(f"missing {e2} from entity dictionary")
                # if rel not in relation_dict: print(f"missing {rel} from relation dictionary")
                continue
            if rel_str in allowed_relations:
                new_data = {
                    'entity1' : entity_dict[e1],
                    'entity2': entity_dict[e2],
                    'relation': relation_dict[rel], 
                }
                all_data.append(new_data)
            # if rel_str not in seen_rels:
            #     seen_rels.append(rel_str)
            #     print(f"{e1_str:45s} | {rel_str:45s} | {e2_str:45s}")
            #     if len(seen_rels) % 10 == 0 or len(seen_rels) == 822:
            #         import pdb; pdb.set_trace()
            # if rel_str in ['']:
            #     print(f"{e1_str:45s} | {rel_str:45s} | {e2_str:45s}")
            #     if len(seen_rels) % 10 == 0 or len(seen_rels) == 822:
            #         import pdb; pdb.set_trace()
            if num % 1000000 == 0 and num != 0:
                print(f"processed {num} points")
print(f"Collected {len(all_data)} points using {len(allowed_relations)} relations")


# want to filter out entities that only have 1 relation
print("Filtering data to have at least 2 relations and combining labels for 1-to-n relations...")
# first make entity info dict
entity_info_dict = {}
for data_num, datapoint in enumerate(all_data):
    e1_str = datapoint['entity1'][0]
    e2_strs = datapoint['entity2']
    rel_str = datapoint['relation'][0]
    e1_strs = datapoint['entity1']
    rel_strs = datapoint['relation']
    if e1_str not in entity_info_dict:
        entity_info_dict[e1_str] = {'rels' : []}
    if rel_str not in entity_info_dict[e1_str]['rels']:
        entity_info_dict[e1_str]['rels'].append(rel_str)

# assign num relations to entity dict
for entity, info in entity_info_dict.items():
    entity_info_dict[entity]['num_rels'] = len(entity_info_dict[entity]['rels'])
    entity_info_dict[entity]['num_unique_rels'] = len(set(entity_info_dict[entity]['rels']))

# get filtered data, add entity idx in filtered_data to entity info dict
filtered_data = []
all_entities = set()
for data_num, datapoint in enumerate(all_data):
    e1_str = datapoint['entity1'][0]
    num_rels = entity_info_dict[e1_str]['num_unique_rels']
    if num_rels > 1:
        filtered_data.append(datapoint)
        all_entities.add(e1_str)
        # add idx of data points in filtered_data
        data_idx = len(filtered_data) - 1
        if 'idx' not in entity_info_dict[e1_str]:
            entity_info_dict[e1_str]['idx'] = [data_idx]
        else:
            entity_info_dict[e1_str]['idx'].append(data_idx)

# concatenate all eligible labels in entity info dict for one-to-many relations
for entity, info in entity_info_dict.items():
    info = entity_info_dict[entity]
    if 'idx' not in info:
        continue
    entity_idx = info['idx']
    all_entity_rels = [filtered_data[idx]['relation'][0] for idx in entity_idx]
    all_entity_entity2 = [filtered_data[idx]['entity2'] for idx in entity_idx]
    unique_rels = info['rels']
    entity_info_dict[entity]['all_rel_labels'] = {}
    for idx, rel in enumerate(unique_rels):
        share_rel_idx = np.argwhere(np.array(all_entity_rels) == rel).reshape(-1)
        all_labels = []
        for share_idx in share_rel_idx:
            all_labels.extend(all_entity_entity2[share_idx])
        entity_info_dict[entity]['all_rel_labels'][rel] = all_labels

# OVERWRITE LABELS BASED ON COMBINED ONE-TO-MANY REL LABELS
for data_num, datapoint in enumerate(all_data):
    e1_str = datapoint['entity1'][0]
    num_rels = entity_info_dict[e1_str]['num_unique_rels']
    rel = datapoint['relation'][0]
    if 'idx' not in entity_info_dict[e1_str]:
        continue
    datapoint['entity2'] = entity_info_dict[e1_str]['all_rel_labels'][rel]

np.random.seed(0)
# np.random.shuffle(filtered_data) # NOT shuffling data after writing the idx above. will shuffle entities and fill in data by entity order
all_entities = list(all_entities)
np.random.shuffle(all_entities)
print(f"Filtered out {len(all_data) - len(filtered_data)} points with only 1 relation")

n_points = len(filtered_data)
n_train = 150000
n_dev = 10000
n_test = 10000
assert n_train + n_dev + n_test < n_points, "not enough points for requested split sizes"

# now need to drop the filtered_data into their respective splits two at a time, to ensure that in every split every entity has at least 2 rels

# make dev/test out of 5000 unique entities with 2 rels each
# make train out of remaining data, as long as at least 2 rels
print("Placing data into splits by sets, so that in every split, every entity is mentioned at least twice with independent relations...")
test_idx = []
dev_idx = []
train_idx = []
current_rel_counts = {rel : 0  for rel in allowed_relations}

for ent_num, entity in enumerate(all_entities):
    info = entity_info_dict[entity]
    if len(test_idx) >= n_test:
        break
    if 'idx' not in info:
        continue
    entity_idx = info['idx']
    relations = filter_for_rare_relations(info['rels'], 
                                    relation_prop_dict = {rel : count / n_test for rel, count in current_rel_counts.items()},
                                    relation_prop_threshold = .11)
    if len(relations) < 2:
        continue
    two_unique_rels = np.random.choice(relations, size=2, replace=False)
    has_rel_one_idx = [idx for idx in entity_idx if filtered_data[idx]['relation'][0] == two_unique_rels[0]]
    has_rel_two_idx = [idx for idx in entity_idx if filtered_data[idx]['relation'][0] == two_unique_rels[1]]
    random_idx1 = np.random.choice(has_rel_one_idx)
    random_idx2 = np.random.choice(has_rel_two_idx)
    test_idx.extend([random_idx1, random_idx2])
    # remove rels from entity_info_dict -- this is what prevents duplicate entity, relation pairs
    entity_info_dict[entity]['rels'] = np.setdiff1d(info['rels'], two_unique_rels)
    for rel in two_unique_rels:
        current_rel_counts[rel] += 1
    progress = len(test_idx)
    if progress % 1000 == 0 and progress != 0:
        print(f"found {progress} test points | {100*ent_num / len(all_entities):.1f}% through entities", end='\r')

# repeat for dev, then train
current_rel_counts = {rel : 0 for rel in allowed_relations}
for ent_num, entity in enumerate(all_entities):
    info = entity_info_dict[entity]
    if len(dev_idx) >= n_dev:
        break
    if 'idx' not in info:
        continue    
    entity_idx = info['idx']
    remaining_rels = info['rels']
    remaining_rels = filter_for_rare_relations(remaining_rels, 
                                relation_prop_dict = {rel : count / n_dev for rel, count in current_rel_counts.items()},
                                relation_prop_threshold = .11)
    if len(remaining_rels) >= 2:
        two_unique_rels = np.random.choice(remaining_rels, size=2, replace=False)
        has_rel_one_idx = [idx for idx in entity_idx if filtered_data[idx]['relation'][0] == two_unique_rels[0]]
        has_rel_two_idx = [idx for idx in entity_idx if filtered_data[idx]['relation'][0] == two_unique_rels[1]]
        random_idx1 = np.random.choice(has_rel_one_idx)
        random_idx2 = np.random.choice(has_rel_two_idx)
        dev_idx.extend([random_idx1, random_idx2])
        for rel in two_unique_rels:
            current_rel_counts[rel] += 1
        # remove rels from entity_info_dict -- this is what prevents duplicate entity, relation pairs
        entity_info_dict[entity]['rels'] = np.setdiff1d(info['rels'], two_unique_rels)
    progress = len(dev_idx)
    if progress % 1000 == 0 and progress != 0:
        print(f"found {progress} dev points | {100*ent_num / len(all_entities):.1f}% through entities", end='\r')

current_rel_counts = {rel : 0 for rel in allowed_relations}
for ent_num, entity in enumerate(all_entities):
    info = entity_info_dict[entity]
    if len(train_idx) >= n_train:
        break
    if 'idx' not in info:
        continue    
    entity_idx = info['idx']
    remaining_rels = info['rels']
    remaining_rels = filter_for_rare_relations(remaining_rels, 
                            relation_prop_dict = {rel : count / n_train for rel, count in current_rel_counts.items()},
                            relation_prop_threshold = .12)
    if len(remaining_rels) >= 2:
        two_unique_rels = np.random.choice(remaining_rels, size=2, replace=False)
        has_rel_one_idx = [idx for idx in entity_idx if filtered_data[idx]['relation'][0] == two_unique_rels[0]]
        has_rel_two_idx = [idx for idx in entity_idx if filtered_data[idx]['relation'][0] == two_unique_rels[1]]
        random_idx1 = np.random.choice(has_rel_one_idx)
        random_idx2 = np.random.choice(has_rel_two_idx)
        train_idx.extend([random_idx1, random_idx2])
        for rel in two_unique_rels:
            current_rel_counts[rel] += 1
    progress = len(train_idx)
    print(f"found {progress} train points | {100*ent_num / len(all_entities):.1f}% through entities", end='\r')


print("Made splits with sizes:")
train_points = [filtered_data[idx] for idx in train_idx]
dev_points = [filtered_data[idx] for idx in dev_idx]
test_points = [filtered_data[idx] for idx in test_idx]
print(f"train: {len(train_points)}")
print(f"dev: {len(dev_points)}")
print(f"test: {len(test_points)}")

# get percent for each relation
train_rels = np.array([point['relation'][0] for point in train_points])
dev_rels = np.array([point['relation'][0] for point in dev_points])
test_rels = np.array([point['relation'][0] for point in test_points])
train_rel_counts = {relation : sum(relation == train_rels) for relation in allowed_relations}
dev_rel_counts = {relation : sum(relation == dev_rels) for relation in allowed_relations}
test_rel_counts = {relation : sum(relation == test_rels) for relation in allowed_relations}

print("Train distr:")
denom = sum(train_rel_counts.values())
for k,v in train_rel_counts.items():
    print(f" {k} : {100*v/denom:.2f}")

print("Dev distr:")
denom = sum(dev_rel_counts.values())
for k,v in dev_rel_counts.items():
    print(f" {k} : {100*v/denom:.2f}")

print("Test distr:")
denom = sum(test_rel_counts.values())
for k,v in test_rel_counts.items():
    print(f" {k} : {100*v/denom:.2f}")

# for dev and test, compute overlap in entities with train
test_overlap = 0
dev_overlap = 0
all_train_entities = [datapoint['entity1'][0] for datapoint in train_points]
print(f"There are {len(set(all_train_entities))} unique entities in the training data")
print("Counting entity overlap with dev and test...")
for point in dev_points:
    entity = point['entity1'][0]
    seen_before = 1*(entity in all_train_entities)
    if seen_before:
        dev_overlap += 1
    point['seen_in_training'] = seen_before
for point in test_points:
    entity = point['entity1'][0]
    seen_before = 1*(entity in all_train_entities)
    if seen_before:
        test_overlap += 1
    point['seen_in_training'] = seen_before

print(f"Dev proportion entities seen at train-time: {100*dev_overlap / n_dev:.2f}")
print(f"Test proportion entities seen at train-time: {100*test_overlap / n_test:.2f}")

# write data
for save_path, data in zip(save_paths, [train_points, dev_points, test_points]):
    with jsonlines.open(save_path, 'w') as file:
        for line in data:
            file.write(line)

print(f"\n Overall runtime: {(time.time() - start) / 3600:.2f} hours")



