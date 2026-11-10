"""Tests for the tokforge Python package.

The WordPiece fixture tests always run. GPT-2 tests compare against tiktoken and run only when
models/gpt2.tokenizer.json (gitignored, fetched with models/fetch.py) and tiktoken are present.
"""

from pathlib import Path

import pytest

import tokforge

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "wordpiece.json"
GPT2 = ROOT / "models" / "gpt2.tokenizer.json"
CORPUS = sorted((ROOT / "tests" / "fixtures" / "corpus").glob("*.txt"))


@pytest.fixture(scope="module")
def wordpiece():
    return tokforge.Tokenizer.from_file(str(FIXTURE))


def test_known_ids_count_and_round_trip(wordpiece):
    assert wordpiece.count("hello world!") == 3
    assert wordpiece.encode("hello world!") == [4, 5, 6]
    assert wordpiece.decode([4, 5, 6]) == "hello world!"


def test_properties(wordpiece):
    assert wordpiece.vocab_size > 0
    assert len(wordpiece.sha256) == 64
    assert wordpiece.core == "backend"
    assert tokforge.__version__


def test_from_str_matches_from_file(wordpiece):
    other = tokforge.Tokenizer.from_str(FIXTURE.read_text(encoding="utf-8"))
    assert other.sha256 == wordpiece.sha256
    assert other.encode("hello world!") == wordpiece.encode("hello world!")


def test_empty_inputs(wordpiece):
    assert wordpiece.encode("") == []
    assert wordpiece.count("") == 0
    assert wordpiece.chunk("", 5) == []
    assert wordpiece.encode_batch([]) == []
    assert wordpiece.count_batch([]) == []
    assert wordpiece.truncate("", 5) == ""


def test_truncate_and_chunk_return_source_slices(wordpiece):
    text = "hello world! hello world!"
    assert text.startswith(wordpiece.truncate(text, 2))
    chunks = wordpiece.chunk(text, 2)
    assert "".join(chunks) == text
    assert all(wordpiece.count(c) <= 2 for c in chunks)


def test_batch_keeps_input_order(wordpiece):
    texts = [f"hello {'world ' * (i % 7)}!" for i in range(200)]
    assert wordpiece.encode_batch(texts) == [wordpiece.encode(t) for t in texts]
    assert wordpiece.count_batch(texts) == [wordpiece.count(t) for t in texts]


def test_lone_surrogate_raises_value_error(wordpiece):
    with pytest.raises(ValueError):
        wordpiece.encode("\ud800")


def test_unknown_id_raises_value_error(wordpiece):
    with pytest.raises(ValueError):
        wordpiece.decode([10**9])


def test_bad_option_raises_value_error():
    with pytest.raises(ValueError):
        tokforge.Tokenizer.from_file(str(FIXTURE), special_tokens="sometimes")
    with pytest.raises(ValueError):
        tokforge.Tokenizer.from_file(str(FIXTURE), core="fastest")


def test_missing_file_raises_os_error():
    with pytest.raises(OSError):
        tokforge.Tokenizer.from_file(str(ROOT / "does-not-exist.json"))


needs_gpt2 = pytest.mark.skipif(not GPT2.exists(), reason="models/gpt2.tokenizer.json not fetched")


@needs_gpt2
def test_gpt2_matches_tiktoken_on_corpus():
    tiktoken = pytest.importorskip("tiktoken")
    reference = tiktoken.get_encoding("gpt2")
    tok = tokforge.Tokenizer.from_file(str(GPT2))
    assert tok.core != "backend"
    for path in CORPUS:
        text = path.read_text(encoding="utf-8")
        assert tok.encode(text) == reference.encode(text, disallowed_special=()), path.name
        assert tok.count(text) == len(reference.encode(text, disallowed_special=())), path.name


@needs_gpt2
def test_gpt2_batch_equals_sequential():
    tok = tokforge.Tokenizer.from_file(str(GPT2))
    texts = [p.read_text(encoding="utf-8") for p in CORPUS] * 20
    assert tok.encode_batch(texts) == [tok.encode(t) for t in texts]
