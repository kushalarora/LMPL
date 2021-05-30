def join_src_tgt(split):
    with open(f'{split}.de') as fsrc, \
        open(f'{split}.en') as ftgt, \
        open(f'{split}.tsv', 'w') as fout:
        
        srcs = [line.strip() for line in fsrc] 
        tgts = [line.strip() for line in ftgt]

        for src, tgt in zip(srcs, tgts):
            fout.write(f'{src}\t{tgt}\n')

print('#' * 40 + 'Train' + '#' * 40)
join_src_tgt('train')
print('#' * 40 + 'Dev  ' + '#' * 40)
join_src_tgt('valid')
print('#' * 40 + 'Test ' + '#' * 40)
join_src_tgt('test')
