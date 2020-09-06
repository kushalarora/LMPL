import sys
import random

train_filename = sys.argv[1]
prefix, suffix = train_filename.split('.')
shards = []
for i in range(1,5):
    shard_filename =f'{prefix}_{i}.{suffix}'
    print(shard_filename)
    shards.append(open(shard_filename, 'w'))

with open(sys.argv[1], 'r') as train_file:
    for line in train_file:
        rndno = random.random()
        shard_idx = 3
        if  rndno < 0.25:
            shard_idx = 0
        elif rndno < 0.50:
            shard_idx = 1
        elif rndno < 0.75:
            shard_idx = 2
        shards[shard_idx].write(line)
