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
    for i,line in enumerate(train_file):
        rndno = random.random()
        shard_idx = 3
        if  (i + 3) % 4 == 0:
            shard_idx = 0
        elif (i + 2) % 4 == 0:
            shard_idx = 1
        elif (i + 1) % 4 == 0:
            shard_idx = 2
        shards[shard_idx].write(line)
