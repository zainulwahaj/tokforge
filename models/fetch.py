"""Download the pinned tokenizer artifacts listed in models/MANIFEST.json.

Usage (from the repository root, isolated mode keeps downloaded files off sys.path):

    python -I models/fetch.py                    # download missing files, verify recorded hashes
    python -I models/fetch.py --artifact gpt2    # one artifact only (repeatable)
    python -I models/fetch.py --write-manifest   # also record sha256/vocab/type fields from the bytes
    python -I models/fetch.py --verify           # verify existing files only, download nothing

Files are fetched from https://huggingface.co/<repo_id>/resolve/<revision>/<remote_file> at the
immutable revision commit recorded in the manifest. Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "MANIFEST.json")
USER_AGENT = "fast-tokenizer-fetch/1 (+https://github.com/zainulwahaj/fast-tokenizer)"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: str, expected_size: int | None) -> None:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        fd, partial = tempfile.mkstemp(dir=os.path.dirname(destination), suffix=".part")
        written = 0
        try:
            with os.fdopen(fd, "wb") as out:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
                    written += len(block)
                    if expected_size is not None and written > expected_size:
                        raise RuntimeError(f"{url}: exceeded expected size {expected_size}")
            if expected_size is not None and written != expected_size:
                raise RuntimeError(f"{url}: got {written} bytes, expected {expected_size}")
            os.replace(partial, destination)
        finally:
            if os.path.exists(partial):
                os.remove(partial)


def describe_component(component) -> str | None:
    """Short structural description of a tokenizer.json component, e.g. Sequence[Split, ByteLevel]."""
    if component is None:
        return None
    kind = component.get("type", "?")
    inner = component.get("normalizers") or component.get("pretokenizers") or component.get("processors") or component.get("decoders")
    if kind == "Sequence" and inner is not None:
        return f"Sequence[{', '.join(describe_component(part) or '?' for part in inner)}]"
    return kind


def describe_model(model: dict) -> str:
    """Model type; older exports (e.g. GPT-2's 2020 tokenizer.json) omit `type`, so infer it from the fields."""
    if "type" in model:
        return model["type"]
    if "merges" in model:
        return "BPE (untagged; inferred from merges)"
    if "continuing_subword_prefix" in model and "max_input_chars_per_word" in model:
        return "WordPiece (untagged; inferred from fields)"
    if isinstance(model.get("vocab"), list):
        return "Unigram (untagged; inferred from scored vocab)"
    return "unknown"


def vocab_size_with_added(data: dict) -> int:
    model = data["model"]
    vocab = model["vocab"]
    if isinstance(vocab, dict):
        keys = set(vocab)
    else:  # Unigram: [[piece, score], ...]
        keys = {piece for piece, _score in vocab}
    extra = sum(1 for token in data.get("added_tokens", []) if token["content"] not in keys)
    return len(keys) + extra


def resolve_url(entry: dict, remote_file: str) -> str:
    return f"https://huggingface.co/{entry['repo_id']}/resolve/{entry['revision']}/{remote_file}"


def process_file(spec: dict, url: str, local: str, args: argparse.Namespace) -> bool:
    """Ensure `local` exists with the expected size and hash. Returns False on a verification failure."""
    label = os.path.relpath(local, HERE)
    if not os.path.exists(local):
        if args.verify:
            print(f"MISSING  {label}")
            return False
        print(f"GET      {label}  <- {url}")
        download(url, local, spec.get("size_bytes"))
    actual_size = os.path.getsize(local)
    actual_sha = sha256_file(local)
    ok = True
    if spec.get("size_bytes") is not None and actual_size != spec["size_bytes"]:
        print(f"SIZE     {label}: {actual_size} != {spec['size_bytes']}")
        ok = False
    if spec.get("sha256"):
        if actual_sha != spec["sha256"]:
            print(f"SHA256   {label}: {actual_sha} != {spec['sha256']}")
            ok = False
        else:
            print(f"OK       {label}  sha256={actual_sha}")
    else:
        print(f"UNPINNED {label}  sha256={actual_sha}")
        if args.write_manifest:
            spec["sha256"] = actual_sha
            spec["size_bytes"] = actual_size
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", action="append", help="artifact name from the manifest (repeatable)")
    parser.add_argument("--write-manifest", action="store_true", help="record hashes and tokenizer types in MANIFEST.json")
    parser.add_argument("--verify", action="store_true", help="only verify files that exist; never download")
    args = parser.parse_args()

    with open(MANIFEST, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    selected = [a for a in manifest["artifacts"] if not args.artifact or a["name"] in args.artifact]
    if args.artifact and len(selected) != len(set(args.artifact)):
        known = ", ".join(a["name"] for a in manifest["artifacts"])
        print(f"unknown artifact; known: {known}", file=sys.stderr)
        return 2

    all_ok = True
    for entry in selected:
        local = os.path.join(HERE, entry["file"])
        ok = process_file(entry, entry["source_url"], local, args)
        all_ok &= ok
        if os.path.exists(local) and args.write_manifest:
            with open(local, "rb") as handle:
                data = json.load(handle)
            entry["vocab_size"] = vocab_size_with_added(data)
            entry["model_type"] = describe_model(data["model"])
            entry["pre_tokenizer"] = describe_component(data.get("pre_tokenizer"))
            entry["normalizer"] = describe_component(data.get("normalizer"))
            entry["post_processor"] = describe_component(data.get("post_processor"))
            entry["decoder"] = describe_component(data.get("decoder"))
        for extra in entry.get("oracle_files", []):
            extra_local = os.path.join(HERE, extra["file"])
            all_ok &= process_file(extra, resolve_url(entry, extra["remote_file"]), extra_local, args)

    if args.write_manifest:
        with open(MANIFEST, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(f"wrote {os.path.relpath(MANIFEST)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
