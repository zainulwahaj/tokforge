# Model artifacts

Place a Hugging Face `tokenizer.json` here and pass its path with
`--tokenizer models/<file>.json`. `tokforge info --tokenizer <path>` reports the
exact artifact SHA-256 and vocabulary size. All `*.json` files in this directory
except `MANIFEST.json` are ignored by Git, as is `oracle/`; record each model's
source and revision in the manifest rather than relying on file names.

## Reference artifacts and MANIFEST.json

[`MANIFEST.json`](MANIFEST.json) pins the four production artifacts that
`tests/reference.rs` checks against independently generated golden IDs:

| Name | Repository (pinned revision) | Family | License |
| --- | --- | --- | --- |
| `gpt2` | `openai-community/gpt2` @ `607a30d7` | byte-level BPE, no normalizer | MIT |
| `qwen2.5-0.5b` | `Qwen/Qwen2.5-0.5B` @ `060db649` | byte-level BPE, Split regex, NFC | Apache-2.0 |
| `bert-base-uncased` | `google-bert/bert-base-uncased` @ `86b5e093` | WordPiece, BERT normalizer | Apache-2.0 |
| `xlm-roberta-base` | `FacebookAI/xlm-roberta-base` @ `e73636d4` | Unigram, precompiled charsmap | MIT |

For each artifact the manifest records the repository id, the immutable revision
commit, the remote and local file names, the resolve URL, the license, the byte
size, the SHA-256 of the exact bytes (identical to what `tokenizer info`
reports), the vocabulary size including added tokens, the model type, and the
pre-tokenizer, normalizer, post-processor, and decoder types. It also states
whether decode is an exact round trip and names the oracle used for the goldens.
Two artifacts (GPT-2 and XLM-R) predate tagged model sections, so their model
type is inferred from the fields and labelled as such.

`oracle_files` lists the extra upstream files the golden generator needs
(`vocab.txt`, `sentencepiece.bpe.model`, `tokenizer_config.json`). They are
downloaded into `models/oracle/<name>/` and are also gitignored.

## Fetching

```powershell
python -I models/fetch.py                     # download anything missing, verify sizes and hashes
python -I models/fetch.py --artifact gpt2     # one artifact (repeatable)
python -I models/fetch.py --verify            # verify only, never download
python -I models/fetch.py --write-manifest    # re-record hashes and types after changing a pin
```

The script uses only the Python standard library, downloads from
`https://huggingface.co/<repo>/resolve/<revision>/<file>`, writes to a temporary
file first, and fails on any size or SHA-256 mismatch. `-I` keeps the download
directory off `sys.path`. Nothing is downloaded automatically by the engine, the
tests, or the benchmark runner.

With the artifacts present, `cargo test --test reference` runs the golden
comparisons and budget invariants; without them those tests print `SKIPPED` and
pass. See `tests/fixtures/golden/generate.py` for how goldens are produced.

## Synthetic fixture

The hand-authored `tests/fixtures/wordpiece.json` is a tiny synthetic test model,
not a production model. It deliberately contains saved padding and truncation
settings so tests verify that the engine disables them.
