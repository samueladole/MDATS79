"""Sample N chunks from the BM25 corpus, one file each, named by chunk_id."""
import json, pathlib, sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
SRC = pathlib.Path("/Users/samuel/Documents/GitHub/MDATS79/data/bm25_corpus.jsonl")
OUT = pathlib.Path(__file__).parent / "docs"
OUT.mkdir(exist_ok=True)
for f in OUT.glob("*.md"):
    f.unlink()

picked = []
with SRC.open() as fh:
    for i, line in enumerate(fh):
        if i % 977 == 0:                      # prime stride: unbiased spread
            r = json.loads(line)
            if 200 < len(r["text"]) < 2000:
                picked.append(r)
        if len(picked) == N:
            break

for r in picked:
    (OUT / f"{r['id'].replace('::', '__')}.md").write_text(r["text"])

lens = [len(r["text"]) for r in picked]
print(f"{len(picked)} chunks -> {OUT}")
print(f"  chars: mean {sum(lens)//len(lens)}, min {min(lens)}, max {max(lens)}")
