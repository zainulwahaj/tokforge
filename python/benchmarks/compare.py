"""Head-to-head throughput: tokforge vs tiktoken vs Hugging Face tokenizers.

Run from the repository root, in an environment with tokforge, tiktoken and tokenizers installed
and the model files fetched (`python -I models/fetch.py`):

    python python/benchmarks/compare.py --output benchmarks/python-compare.json

Every library loads the same tokenizer (tiktoken by encoding name, the others from the same
tokenizer.json). Before timing anything, the script checks that all three return identical ids
for every dataset and stops if they do not. Datasets come from this repository, so the run needs
no downloads beyond the models:

- english: the project's Markdown documentation (technical prose)
- code: the project's Rust sources
- multilingual: tests/fixtures/corpus, the hard-case corpus (many scripts, emoji, controls)

Workloads:
- single: the whole dataset as one string, one thread
- batch: the dataset cut into ~4 KB documents on line boundaries, all cores
  (tiktoken num_threads=os.cpu_count(); tokforge and tokenizers use their rayon pools)

Each figure is the median of --repeat timed runs after one warm-up run, in MB/s of UTF-8 input.

Outside text replaces the built-in datasets with --corpus NAME=PATH[,PATH...][@MAX_BYTES], repeatable,
for example --corpus wikipedia=data/bench/enwik8@8000000. Files are joined with a newline and cut
to MAX_BYTES on a character boundary. --models points at another directory of tokenizer files.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENCODINGS = ["gpt2", "cl100k_base", "o200k_base"]


def datasets() -> dict[str, str]:
    docs = sorted(p for p in ROOT.rglob("*.md") if not any(part.startswith(".") or part in ("target", "node_modules") for part in p.relative_to(ROOT).parts))
    rust = sorted(p for d in ("src", "tests", "benches", "examples") for p in (ROOT / d).rglob("*.rs"))
    corpus = sorted((ROOT / "tests" / "fixtures" / "corpus").glob("*.txt"))
    read = lambda paths: "\n".join(p.read_text(encoding="utf-8") for p in paths)
    return {"english": read(docs), "code": read(rust), "multilingual": read(corpus)}


def corpus(spec: str) -> tuple[str, str]:
    name, _, rest = spec.partition("=")
    paths, _, limit = rest.partition("@")
    text = "\n".join(Path(p).read_text(encoding="utf-8", errors="replace") for p in paths.split(","))
    if limit:
        text = text.encode("utf-8")[: int(limit)].decode("utf-8", errors="ignore")
    return name, text


def batch_of(text: str, size: int = 4096) -> list[str]:
    docs, current = [], []
    length = 0
    for line in text.splitlines(keepends=True):
        current.append(line)
        length += len(line)
        if length >= size:
            docs.append("".join(current))
            current, length = [], 0
    if current:
        docs.append("".join(current))
    return docs


def median_seconds(run, repeat: int) -> float:
    run()  # warm-up: caches, thread pools, lazy initialisation
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeat", type=int, default=9)
    parser.add_argument("--output")
    parser.add_argument("--corpus", action="append", default=[], help="NAME=PATH[,PATH...][@MAX_BYTES]")
    parser.add_argument("--encodings", default=",".join(ENCODINGS))
    parser.add_argument("--models", default=str(ROOT / "models"))
    args = parser.parse_args()

    import tiktoken
    import tokenizers
    import tokforge

    threads = os.cpu_count() or 1
    data = dict(corpus(c) for c in args.corpus) if args.corpus else datasets()
    batches = {name: batch_of(text) for name, text in data.items()}
    results = []
    encodings = args.encodings.split(",")
    for encoding in encodings:
        path = Path(args.models) / f"{encoding}.tokenizer.json"
        if not path.exists():
            print(f"skip {encoding}: {path} not fetched", file=sys.stderr)
            continue
        tk = tiktoken.get_encoding(encoding)
        tf = tokforge.Tokenizer.from_file(str(path))
        hf = tokenizers.Tokenizer.from_file(str(path))
        hf.encode_special_tokens = True  # special spellings are plain text, as in the others
        libraries = {
            "tokforge": (lambda t: tf.encode(t), lambda b: tf.encode_batch(b)),
            "tiktoken": (
                lambda t: tk.encode(t, disallowed_special=()),
                lambda b: tk.encode_batch(b, disallowed_special=(), num_threads=threads),
            ),
            "tokenizers": (
                lambda t: hf.encode(t, add_special_tokens=False).ids,
                lambda b: [e.ids for e in hf.encode_batch_fast(b, add_special_tokens=False)],
            ),
        }
        for name, text in data.items():
            expected = tk.encode(text, disallowed_special=())
            for library, (single, batch) in libraries.items():
                if single(text) != expected:
                    raise SystemExit(f"{library} disagrees with tiktoken on {encoding}/{name}; not timing")
                if [i for ids in batch(batches[name]) for i in ids] != [i for d in batches[name] for i in tk.encode(d, disallowed_special=())]:
                    raise SystemExit(f"{library} batch disagrees with tiktoken on {encoding}/{name}; not timing")
            megabytes = len(text.encode("utf-8")) / 1e6
            for library, (single, batch) in libraries.items():
                for workload, run in (("single", lambda: single(text)), ("batch", lambda: batch(batches[name]))):
                    seconds = median_seconds(run, args.repeat)
                    results.append({"encoding": encoding, "dataset": name, "workload": workload, "library": library,
                                    "mb_per_s": round(megabytes / seconds, 2), "seconds": seconds,
                                    "bytes": len(text.encode("utf-8")), "tokens": len(expected)})
            print(f"done {encoding}/{name}", file=sys.stderr)

    report = {
        "machine": {"cpu": platform.processor(), "logical_cpus": threads, "os": platform.platform(), "python": platform.python_version()},
        "versions": {"tokforge": tokforge.__version__, "tiktoken": tiktoken.__version__, "tokenizers": tokenizers.__version__},
        "method": __doc__.split("Workloads:")[1].strip(),
        "repeat": args.repeat,
        "datasets": {name: len(text.encode("utf-8")) for name, text in data.items()},
        "results": results,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'encoding':12s} {'dataset':13s} {'workload':8s} {'tokforge':>9s} {'tiktoken':>9s} {'tokenizers':>11s}  vs tiktoken")
    key = lambda r: (r["encoding"], r["dataset"], r["workload"])
    for group in sorted({key(r) for r in results}, key=lambda g: (encodings.index(g[0]), g[1], g[2] != "single")):
        row = {r["library"]: r["mb_per_s"] for r in results if key(r) == group}
        print(f"{group[0]:12s} {group[1]:13s} {group[2]:8s} {row['tokforge']:9.1f} {row['tiktoken']:9.1f} {row['tokenizers']:11.1f}  {row['tokforge'] / row['tiktoken']:5.2f}x")
    print("\nMB/s of UTF-8 input, median of", args.repeat, "runs;", report["machine"]["cpu"], f"({threads} logical CPUs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
