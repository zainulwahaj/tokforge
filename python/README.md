# tokforge

Fast BPE tokenization for Python, backed by a Rust engine.

GPT-2, GPT-4 (`cl100k_base`), GPT-4o (`o200k_base`) and Qwen2-style tokenizers run on a
byte-level BPE core written from scratch, checked against tiktoken and Hugging Face `tokenizers`
by differential tests and long soak runs without a mismatch. On the project's benchmark it is
1.4x to 4.6x as fast as tiktoken with identical output. Any other Hugging Face `tokenizer.json`
(BERT, XLM-R, ...) loads too and runs on the Hugging Face backend.

```python
import tokforge

tok = tokforge.Tokenizer.from_file("tokenizer.json")

tok.encode("hello world")              # [31373, 995]
tok.count("hello world")               # 2
tok.decode([31373, 995])               # "hello world"
tok.encode_batch(["a", "b", "c"])      # parallel, releases the GIL
tok.truncate(long_text, 512)           # a prefix of long_text within 512 tokens
tok.chunk(long_text, 512)              # slices that join back to long_text
```

Options, all keyword-only:

- `special_tokens="literal"` (default) treats spellings like `<|endoftext|>` in the input as
  plain text; `"recognize"` matches them as special tokens.
- `add_special_tokens=False` (default); `True` adds the model's BOS/EOS tokens.
- `core="auto"` (default) uses the custom core where it applies; `"backend"` forces the
  Hugging Face path; `"verify"` runs both and raises if they ever disagree.

Errors raise `ValueError` (bad options, unknown ids, invalid input) or `OSError` (file).

License: MIT OR Apache-2.0.
