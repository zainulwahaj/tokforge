"""Generate golden token IDs for tests/reference.rs from independent oracles.

Usage (repository root; `-I` keeps the model directory off sys.path):

    python -I tests/fixtures/golden/generate.py                   # all artifacts whose files exist
    python -I tests/fixtures/golden/generate.py --artifact gpt2   # one artifact (repeatable)
    python -I tests/fixtures/golden/generate.py --check           # regenerate in memory, diff, write nothing

Prerequisites: `python -I models/fetch.py` (artifacts + oracle files at pinned revisions) and the
packages below installed in the interpreter that runs this script (a venv is fine):

    pip install tiktoken sentencepiece transformers tokenizers

Oracles per artifact (recorded in every golden file):
  gpt2              tiktoken `gpt2` encoding (independent implementation of GPT-2 byte-level BPE)
  qwen2.5-0.5b      transformers fast tokenizer (AutoTokenizer on the pinned tokenizer.json +
                    tokenizer_config.json). NOT independent: it wraps the Hugging Face tokenizers
                    Rust backend, the same family the engine uses; this is a cross-version check.
  bert-base-uncased transformers slow BertTokenizer (pure-Python port of Google's tokenization.py)
  xlm-roberta-base  Google sentencepiece on the original sentencepiece.bpe.model for literal mode;
                    transformers slow XLMRobertaTokenizer for recognize mode and special tokens.

Every case is additionally cross-checked against the `tokenizers` Python bindings loaded from the
exact pinned tokenizer.json; the result is recorded, never used to alter the oracle's IDs.
Literal mode treats special-token spellings in text as ordinary text; recognize mode extracts them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MODELS = os.path.join(ROOT, "models")
CORPUS = os.path.join(ROOT, "tests", "fixtures", "corpus")
GENERATOR = "tests/fixtures/golden/generate.py"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_corpus() -> list[tuple[str, bytes, str]]:
    files = sorted(name for name in os.listdir(CORPUS) if name.endswith(".txt"))
    out = []
    for name in files:
        with open(os.path.join(CORPUS, name), "rb") as handle:
            data = handle.read()
        out.append((name, data, data.decode("utf-8")))  # strict: corpus must be valid UTF-8
    return out


class Oracle:
    """Produces IDs for (literal: bool, add_special_tokens: bool). Subclasses set `name`."""

    name = "?"
    adds_special_tokens = False

    def cases(self) -> list[tuple[bool, bool]]:
        cases = [(True, False), (False, False)]
        if self.adds_special_tokens:
            cases += [(True, True), (False, True)]
        return cases

    def encode(self, text: str, literal: bool, add_special_tokens: bool) -> list[int]:
        raise NotImplementedError


class TiktokenGpt2(Oracle):
    def __init__(self) -> None:
        import tiktoken

        self.enc = tiktoken.get_encoding("gpt2")
        self.name = f"tiktoken {tiktoken.__version__} encoding=gpt2"

    def encode(self, text, literal, add_special_tokens):
        assert not add_special_tokens
        if literal:
            return self.enc.encode(text, disallowed_special=())
        return self.enc.encode(text, allowed_special="all")


class TransformersFast(Oracle):
    """transformers fast tokenizer; shares the HF Rust backend family with the engine."""

    def __init__(self, tokenizer_json: str, tokenizer_config: str) -> None:
        import tokenizers
        import transformers

        self.tmp = tempfile.mkdtemp(prefix="fast-tokenizer-oracle-")
        shutil.copy(tokenizer_json, os.path.join(self.tmp, "tokenizer.json"))
        shutil.copy(tokenizer_config, os.path.join(self.tmp, "tokenizer_config.json"))
        self.tok = transformers.AutoTokenizer.from_pretrained(self.tmp)
        assert self.tok.is_fast, "expected a fast tokenizer"
        self.adds_special_tokens = self.tok.num_special_tokens_to_add() > 0
        self.name = (
            f"transformers {transformers.__version__} {type(self.tok).__name__} "
            f"(tokenizers {tokenizers.__version__} Rust backend, not independent)"
        )

    def encode(self, text, literal, add_special_tokens):
        return self.tok.encode(text, add_special_tokens=add_special_tokens, split_special_tokens=literal)


class SlowTransformers(Oracle):
    """A pure-Python (slow) transformers tokenizer; `tokenize(split_special_tokens=...)` gives both modes."""

    def __init__(self, tok, label: str) -> None:
        import transformers

        assert not tok.is_fast, f"{label} must be a slow tokenizer to be independent of the Rust backend"
        self.tok = tok
        self.adds_special_tokens = tok.num_special_tokens_to_add() > 0
        self.name = f"transformers {transformers.__version__} {type(tok).__name__} (slow, {label})"

    def encode(self, text, literal, add_special_tokens):
        tokens = self.tok.tokenize(text, split_special_tokens=literal)
        ids = self.tok.convert_tokens_to_ids(tokens)
        if add_special_tokens:
            ids = self.tok.build_inputs_with_special_tokens(ids)
        return ids


class SlowBert(SlowTransformers):
    """Literal mode runs BasicTokenizer then WordpieceTokenizer directly, i.e. Google's original
    FullTokenizer.tokenize loop; the wrapper's `split_special_tokens` still keeps `[CLS]` whole via
    BasicTokenizer's never_split in transformers 4.57. Recognize mode is the wrapper's tokenize()."""

    def __init__(self, vocab_txt: str) -> None:
        import transformers

        tok = transformers.BertTokenizer(vocab_file=vocab_txt, do_lower_case=True)
        super().__init__(tok, "BasicTokenizer+WordpieceTokenizer from Google tokenization.py")
        assert tok.basic_tokenizer.do_lower_case and tok.basic_tokenizer.strip_accents is None
        assert not tok.basic_tokenizer.never_split
        assert self._literal_tokens("[CLS] x") == ["[", "cl", "##s", "]", "x"]
        assert self.tok.tokenize("[CLS] x") == ["[CLS]", "x"]

    def _literal_tokens(self, text: str) -> list[str]:
        return [
            piece
            for word in self.tok.basic_tokenizer.tokenize(text)
            for piece in self.tok.wordpiece_tokenizer.tokenize(word)
        ]

    def encode(self, text, literal, add_special_tokens):
        if not literal:
            return super().encode(text, literal, add_special_tokens)
        ids = self.tok.convert_tokens_to_ids(self._literal_tokens(text))
        return self.tok.build_inputs_with_special_tokens(ids) if add_special_tokens else ids


class SentencePieceXlmr(Oracle):
    """Raw sentencepiece for literal mode; slow XLMRobertaTokenizer for recognize/special tokens."""

    def __init__(self, spm_model: str) -> None:
        import sentencepiece
        import transformers

        self.sp = sentencepiece.SentencePieceProcessor(model_file=spm_model)
        assert self.sp.unk_id() == 0 and self.sp.bos_id() == 1 and self.sp.eos_id() == 2
        self.slow = transformers.XLMRobertaTokenizer(vocab_file=spm_model)
        assert not self.slow.is_fast
        self.adds_special_tokens = True
        self.spm_name = f"sentencepiece {sentencepiece.__version__} (fairseq offset: spm id k>=3 -> k+1, <unk> 0 -> 3)"
        self.slow_name = f"transformers {transformers.__version__} XLMRobertaTokenizer (slow, sentencepiece-backed)"
        self.name = f"{self.spm_name}; {self.slow_name}"
        # The slow tokenizer must agree with the raw model plus offset, or the attribution is wrong.
        probe = "Hello world, 你好 <s> test"
        assert self.encode(probe, True, False) == self._slow(probe, True, False)

    def _spm(self, text: str) -> list[int]:
        return [3 if i == 0 else i + 1 for i in self.sp.encode(text, out_type=int)]

    def _slow(self, text: str, literal: bool, add_special_tokens: bool) -> list[int]:
        ids = self.slow.convert_tokens_to_ids(self.slow.tokenize(text, split_special_tokens=literal))
        return self.slow.build_inputs_with_special_tokens(ids) if add_special_tokens else ids

    def encode(self, text, literal, add_special_tokens):
        if literal and not add_special_tokens:
            return self._spm(text)
        return self._slow(text, literal, add_special_tokens)

    def oracle_for(self, literal: bool, add_special_tokens: bool) -> str:
        return self.spm_name if literal and not add_special_tokens else self.slow_name


class CrossCheck:
    """`tokenizers` Python bindings on the exact pinned tokenizer.json (same backend family as the engine)."""

    def __init__(self, tokenizer_json: str) -> None:
        import tokenizers

        self.tok = tokenizers.Tokenizer.from_file(tokenizer_json)
        self.tok.no_truncation()
        self.tok.no_padding()
        self.name = f"tokenizers {tokenizers.__version__} (Python bindings of the HF Rust backend)"

    def encode(self, text: str, literal: bool, add_special_tokens: bool) -> list[int]:
        self.tok.encode_special_tokens = literal
        return self.tok.encode(text, add_special_tokens=add_special_tokens).ids


def build_oracle(entry: dict) -> Oracle:
    name = entry["name"]
    model = os.path.join(MODELS, entry["file"])
    extras = {e["remote_file"]: os.path.join(MODELS, e["file"]) for e in entry.get("oracle_files", [])}
    if name == "gpt2":
        return TiktokenGpt2()
    if name == "qwen2.5-0.5b":
        return TransformersFast(model, extras["tokenizer_config.json"])
    if name == "bert-base-uncased":
        return SlowBert(extras["vocab.txt"])
    if name == "xlm-roberta-base":
        return SentencePieceXlmr(extras["sentencepiece.bpe.model"])
    raise SystemExit(f"no oracle defined for artifact {name}")


# Header fields that describe the machine that ran the generator, not the result. A different
# Python patch release cannot change token ids, so --check leaves these out of the comparison;
# the oracle and cross-check versions (which can) stay in it.
ENVIRONMENT_FIELDS = ("python",)


def comparable(text: str) -> dict:
    data = json.loads(text)
    for field in ENVIRONMENT_FIELDS:
        data.pop(field, None)
    return data


def render(golden: dict) -> str:
    """Pretty JSON with every `ids` array kept on a single line."""
    placeholders = {}
    for index, case in enumerate(golden["cases"]):
        for key in ("ids", "cross_check_ids"):
            if key in case:
                token = f"@@{key}{index}@@"
                placeholders[f'"{token}"'] = json.dumps(case[key], separators=(",", ":"))
                case[key] = token
    text = json.dumps(golden, indent=2, ensure_ascii=True)
    for token, replacement in placeholders.items():
        text = text.replace(token, replacement)
    return text + "\n"


def generate(entry: dict, corpus, args) -> int:
    name = entry["name"]
    model = os.path.join(MODELS, entry["file"])
    with open(model, "rb") as handle:
        artifact_sha = sha256_bytes(handle.read())
    if entry.get("sha256") and entry["sha256"] != artifact_sha:
        raise SystemExit(f"{name}: {model} sha256 {artifact_sha} != manifest {entry['sha256']}")
    oracle = build_oracle(entry)
    cross = CrossCheck(model)
    out_dir = os.path.join(HERE, name)
    os.makedirs(out_dir, exist_ok=True)
    differences = 0
    for corpus_name, data, text in corpus:
        cases = []
        for literal, add_special in oracle.cases():
            ids = [int(i) for i in oracle.encode(text, literal, add_special)]
            cross_ids = cross.encode(text, literal, add_special)
            case = {
                "special_tokens": "literal" if literal else "recognize",
                "add_special_tokens": add_special,
                "oracle": oracle.oracle_for(literal, add_special) if hasattr(oracle, "oracle_for") else oracle.name,
                "ids": ids,
                "cross_check_matches": cross_ids == ids,
            }
            if cross_ids != ids:
                case["cross_check_ids"] = cross_ids
            cases.append(case)
        golden = {
            "schema_version": 1,
            "artifact": name,
            "artifact_sha256": artifact_sha,
            "corpus_file": corpus_name,
            "corpus_sha256": sha256_bytes(data),
            "corpus_bytes": len(data),
            "text_is_nfc": unicodedata.is_normalized("NFC", text),
            "generator": GENERATOR,
            "python": platform.python_version(),
            "oracle": oracle.name,
            "cross_check": cross.name,
            "cases": cases,
        }
        rendered = render(golden)
        target = os.path.join(out_dir, corpus_name[: -len(".txt")] + ".json")
        if args.check:
            existing = open(target, "r", encoding="utf-8").read() if os.path.exists(target) else None
            same = existing is not None and comparable(existing) == comparable(rendered)
            status = "same" if same else "DIFFERENT"
            differences += status != "same"
            print(f"{status:9s} {os.path.relpath(target, ROOT)}")
        else:
            with open(target, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(rendered)
            mismatched = sum(1 for c in cases if not c["cross_check_matches"])
            print(f"wrote {os.path.relpath(target, ROOT)}  cases={len(cases)} cross_check_mismatches={mismatched}")
    return differences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", action="append")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    with open(os.path.join(MODELS, "MANIFEST.json"), "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    corpus = read_corpus()
    differences = 0
    for entry in manifest["artifacts"]:
        if args.artifact and entry["name"] not in args.artifact:
            continue
        if not os.path.exists(os.path.join(MODELS, entry["file"])):
            print(f"skip {entry['name']}: {entry['file']} not downloaded")
            continue
        differences += generate(entry, corpus, args)
    return 1 if differences else 0


if __name__ == "__main__":
    sys.exit(main())
