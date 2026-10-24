"""Sample N chunks from the BM25 corpus, one file each, named by chunk_id.

Sampling is corpus-principled: an even stride across the whole corpus, with no
reference to the benchmark queries. A subset seeded from gold passages would
produce an oracle graph that flatters any later retrieval comparison.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from pipeline_config import CORPUS_PATH as SRC  # noqa: E402
from pipeline_config import SAMPLE_DIR as OUT  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
OUT.mkdir(parents=True, exist_ok=True)
for f in OUT.glob("*.md"):
    f.unlink()

total = sum(1 for _ in SRC.open())
# oversample ~1.6x to survive the length filter, then trim to exactly N
stride = max(1, total // int(N * 1.6))

picked = []
with SRC.open() as fh:
    for i, line in enumerate(fh):
        if i % stride == 0:
            record = json.loads(line)
            if 200 < len(record["text"]) < 2000:
                picked.append(record)
        if len(picked) >= N:
            break

picked = picked[:N]
for record in picked:
    name = record["id"].replace("::", "__")
    (OUT / f"{name}.md").write_text(record["text"])

lens = [len(r["text"]) for r in picked]
print(f"{len(picked)} chunks -> {OUT}  (corpus {total:,}, stride {stride})")
print(f"  chars: mean {sum(lens)//len(lens)}, min {min(lens)}, max {max(lens)}")
