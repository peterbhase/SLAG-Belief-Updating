import os
import jsonlines
import numpy as np
import math
from copy import deepcopy


if os.environ.get('DATA_DIR') is not None:
    data_dir = os.environ.get('DATA_DIR')
else:
    data_dir = 'data/'

train_path = os.path.join(data_dir, 'KILT', 'fever-train-kilt.jsonl')
dev_path = os.path.join(data_dir, 'KILT', 'fever-dev-kilt.jsonl')

train_points = list(jsonlines.open(train_path))
dev_points = list(jsonlines.open(dev_path))

print(f"train prop true: ", round(np.mean([point["output"][0]["answer"] == 'SUPPORTS' for point in train_points]),2) )
print(f"dev prop true: ", round(np.mean([point["output"][0]["answer"] == 'SUPPORTS' for point in dev_points]),2) )

np.random.seed(1)
np.random.shuffle(train_points)

n_points = len(train_points)
n_train = math.ceil(.9 * n_points)
n_dev = math.ceil(.1 * n_points)

while n_train + n_dev > n_points:
    n_train -= 1

# backwards order for reassignments
test_points = deepcopy(dev_points)
dev_points = deepcopy(train_points[n_train+1:])
train_points = deepcopy(train_points[:n_train])

print("train size: ", len(train_points))
print("dev size: ", len(dev_points))
print("test size: ", len(test_points))
print(f"train prop true: ", round(np.mean([point["output"][0]["answer"] == 'SUPPORTS' for point in train_points]),2) )
print(f"dev prop true: ", round(np.mean([point["output"][0]["answer"] == 'SUPPORTS' for point in dev_points]),2) )
print(f"test prop true: ", round(np.mean([point["output"][0]["answer"] == 'SUPPORTS' for point in test_points]),2) )

train_path = os.path.join(data_dir, 'KILT', 'fever_reshuffled-train-kilt.jsonl')
dev_path = os.path.join(data_dir, 'KILT', 'fever_reshuffled-dev-kilt.jsonl')
test_path = os.path.join(data_dir, 'KILT', 'fever_reshuffled-test-kilt.jsonl')

with jsonlines.open(train_path, mode='w') as file:
    for point in train_points:
        file.write(point)

with jsonlines.open(dev_path, mode='w') as file:
    for point in dev_points:
        file.write(point)

with jsonlines.open(test_path, mode='w') as file:
    for point in test_points:
        file.write(point)