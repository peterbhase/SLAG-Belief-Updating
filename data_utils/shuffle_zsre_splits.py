import os
import jsonlines
import numpy as np
import math

if os.environ.get('DATA_DIR') is not None:
    data_dir = os.environ.get('DATA_DIR')
else:
    data_dir = 'data/'
    
train_path = os.path.join(data_dir, 'KILT', 'structured_zeroshot-train-kilt.jsonl')
dev_path = os.path.join(data_dir, 'KILT', 'structured_zeroshot-dev-kilt.jsonl')

train_points = list(jsonlines.open(train_path))
dev_points = list(jsonlines.open(dev_path))
    
combined_data = train_points + dev_points

np.random.seed(0)
np.random.shuffle(combined_data)

n_points = len(combined_data)
n_train = math.ceil(.8 * n_points)
n_dev = math.ceil(.1 * n_points)
n_test = math.ceil(.1 * n_points)

while n_train + n_dev + n_test > n_points:
    n_train -= 1

train_points = combined_data[:n_train]
dev_points = combined_data[n_train+1:(n_train+n_dev)]
test_points = combined_data[n_train+n_dev+1:]

# check how many points have paraphrases in each
n_train_paraphrases = 0
n_dev_paraphrases = 0
n_test_paraphrases = 0 
for point in train_points:
    if len(point["meta"]['template_questions']) > 1:
        n_train_paraphrases += 1
for point in dev_points:
    if len(point["meta"]['template_questions']) > 1:
        n_dev_paraphrases += 1
for point in test_points:
    if len(point["meta"]['template_questions']) > 1:
        n_test_paraphrases += 1

print("train size: ", len(train_points))
print("train points w/ paraphrases: ", n_train_paraphrases)
print("dev size: ", len(dev_points))
print("dev points w/ paraphrases: ", n_dev_paraphrases)
print("test size: ", len(test_points))
print("test points w/ paraphrases: ", n_test_paraphrases)

train_path = os.path.join(data_dir, 'KILT', 'zeroshot_reshuffled-train-kilt.jsonl')
dev_path = os.path.join(data_dir, 'KILT', 'zeroshot_reshuffled-dev-kilt.jsonl')
test_path = os.path.join(data_dir, 'KILT', 'zeroshot_reshuffled-test-kilt.jsonl')

with jsonlines.open(train_path, mode='w') as file:
    for point in train_points:
        file.write(point)

with jsonlines.open(dev_path, mode='w') as file:
    for point in dev_points:
        file.write(point)

with jsonlines.open(test_path, mode='w') as file:
    for point in test_points:
        file.write(point)