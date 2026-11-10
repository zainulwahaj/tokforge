"""Memory: tokenizer load and batch peak per library, and the HTTP server's peak at its admission bound.

    python python/benchmarks/memory.py --output benchmarks/memory.json

Library part: each (library, encoding) runs in a fresh subprocess that records resident memory
before and after loading, the process peak reached while loading, and the peak after encoding the
`english` dataset of compare.py as ~4 KB documents in one batch. Peaks are cumulative, so
`encode_peak_over_load_peak_mib` is what encoding added on top of the load-time peak.

Server part: starts `target/release/tokforge serve` (build it first) on GPT-2 with the default 4
workers and 16 queue slots, sends 20 concurrent `POST /v1/encode` requests whose bodies are just
under the default 1 MiB limit, and reads the server's peak memory. That is the admission bound the
docs describe: 20 admitted slots x 1 MiB bodies, plus the encode responses (ids, offsets and token
strings for every token), which are several times larger than the input.

Peaks use the OS-recorded peak working set on Windows (psutil `peak_wset`) and `ru_maxrss` style
peaks elsewhere (psutil `memory_info().rss` sampled), so they do not depend on sampling.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

CHILD = r"""
import json, os, sys, psutil
sys.path.insert(0, sys.argv[4])
from compare import batch_of, datasets
library, encoding, path = sys.argv[1], sys.argv[2], sys.argv[3]
proc = psutil.Process()
mb = lambda b: round(b / 2**20, 1)
peak = lambda: getattr(proc.memory_info(), "peak_wset", None) or proc.memory_info().rss
docs = batch_of(datasets()["english"])
base = proc.memory_info().rss
if library == "tokforge":
    import tokforge; tok = tokforge.Tokenizer.from_file(path); run = lambda: tok.encode_batch(docs)
elif library == "tiktoken":
    import tiktoken; tok = tiktoken.get_encoding(encoding); tok.encode("warm"); run = lambda: tok.encode_batch(docs, disallowed_special=(), num_threads=os.cpu_count())
else:
    import tokenizers; tok = tokenizers.Tokenizer.from_file(path); run = lambda: tok.encode_batch_fast(docs, add_special_tokens=False)
loaded = proc.memory_info().rss
load_peak = peak()
out = run()
print(json.dumps({"base_mib": mb(base), "loaded_mib": mb(loaded), "load_delta_mib": mb(loaded - base),
                  "load_peak_mib": mb(load_peak), "encode_peak_mib": mb(peak()),
                  "encode_peak_over_load_peak_mib": mb(peak() - load_peak)}))
"""


def library_memory() -> list[dict]:
    rows = []
    for encoding in ("gpt2", "cl100k_base", "o200k_base"):
        path = ROOT / "models" / f"{encoding}.tokenizer.json"
        if not path.exists():
            continue
        for library in ("tokforge", "tiktoken", "tokenizers"):
            out = subprocess.run([sys.executable, "-c", CHILD, library, encoding, str(path), str(Path(__file__).resolve().parent)],
                                 capture_output=True, text=True, check=True)
            rows.append({"encoding": encoding, "library": library, **json.loads(out.stdout)})
            print(f"{encoding:12s} {library:10s} {rows[-1]}", file=sys.stderr)
    return rows


def server_memory(port: int = 18093) -> dict:
    import psutil

    exe = ROOT / "target" / "release" / ("tokforge.exe" if sys.platform == "win32" else "tokforge")
    server = subprocess.Popen([str(exe), "serve", "--tokenizer", str(ROOT / "models" / "gpt2.tokenizer.json"),
                               "--bind", f"127.0.0.1:{port}", "--workers", "4", "--queue-capacity", "16"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        proc = psutil.Process(server.pid)
        idle = proc.memory_info().rss
        # English-like text just under the 1 MiB body limit once JSON-encoded.
        prose = (ROOT / "tests" / "fixtures" / "corpus" / "ascii_prose.txt").read_text(encoding="utf-8")
        text = (prose * (1_000_000 // len(prose) + 1))[: 1_000_000]
        body = json.dumps({"text": text}).encode()
        assert len(body) < 2**20

        def send(_):
            request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/encode", data=body,
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.status, len(response.read())

        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(20) as pool:
            replies = list(pool.map(send, range(20)))
        elapsed = time.perf_counter() - start
        info = proc.memory_info()
        peak = getattr(info, "peak_wset", None) or info.rss
        return {"artifact": "gpt2", "workers": 4, "queue_capacity": 16, "concurrent_requests": 20,
                "request_body_bytes": len(body), "statuses": sorted({s for s, _ in replies}),
                "response_bytes_each": replies[0][1], "seconds": round(elapsed, 2),
                "idle_mib": round(idle / 2**20, 1), "peak_mib": round(peak / 2**20, 1)}
    finally:
        server.terminate()
        server.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = {"machine": {"cpu": platform.processor(), "os": platform.platform()},
              "libraries": library_memory(), "server": server_memory()}
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["server"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
