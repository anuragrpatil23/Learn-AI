"""Turn fineweb-edu text into files of GPT-2 token ids for train.py.

Reads the .arrow files that build-nanogpt/fineweb.py downloaded, tokenizes each document with an
end-of-text token in front, and writes shards of uint16 token ids.

Usage: python make_tokens.py '<folder of .arrow files>/*.arrow' <output folder> --shards 2 --size 1e8
"""
import argparse, glob, os
import numpy as np, pyarrow as pa
from transformers import GPT2TokenizerFast

p = argparse.ArgumentParser()
p.add_argument("arrow"); p.add_argument("out")
p.add_argument("--shards", type=int, default=2)
p.add_argument("--size", type=float, default=1e8, help="tokens per shard")
args = p.parse_args()

tok = GPT2TokenizerFast.from_pretrained("gpt2")
os.makedirs(args.out, exist_ok=True)
size, buf, have, shard = int(args.size), [], 0, 0

def texts():
    for path in sorted(glob.glob(args.arrow)):
        for batch in pa.ipc.open_stream(path):
            yield batch.column("text").to_pylist()

for docs in texts():
    for ids in tok(docs)["input_ids"]:
        buf.append(np.array([50256] + ids, dtype=np.uint16)); have += len(ids) + 1
    if have >= size:
        flat = np.concatenate(buf)
        np.save(os.path.join(args.out, "tokens_%03d.npy" % shard), flat[:size])
        print("wrote shard", shard, flush=True)
        buf, have, shard = [flat[size:]], len(flat) - size, shard + 1
        if shard == args.shards:
            break
