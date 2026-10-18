Place your model's exported Hugging Face `tokenizer.json` here and pass its path
using `--tokenizer models/tokenizer.json`. Record the source model and revision
alongside it; `tokenizer info --tokenizer models/tokenizer.json` reports the exact
artifact SHA-256. Large model artifacts are ignored by Git.

The hand-authored `tests/fixtures/wordpiece.json` is a tiny synthetic test model,
not a production model. It deliberately contains saved padding and truncation
settings so tests verify that the engine disables them.
