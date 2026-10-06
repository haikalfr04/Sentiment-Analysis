import pytest
from transformers import BertTokenizerFast

from src.data import clean_text, encode, head_tail_truncate

WORDS = [f"w{i}" for i in range(100)]


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    vocab = tmp_path_factory.mktemp("tok") / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]))
    # Positional so it works with both transformers v4 (vocab_file) and v5 (vocab).
    return BertTokenizerFast(str(vocab))


def test_clean_text_removes_br_tags_and_extra_whitespace():
    assert clean_text("Great movie.<br /><br />Loved it.  <BR>Really") == "Great movie. Loved it. Really"


def test_head_tail_keeps_short_texts_unchanged(tokenizer):
    text = " ".join(WORDS[:10])
    assert head_tail_truncate(tokenizer, [text], max_length=32) == [text]


def test_head_tail_keeps_start_and_end_of_long_texts(tokenizer):
    text = " ".join(WORDS)  # 100 tokens
    (short,) = head_tail_truncate(tokenizer, [text], max_length=22)  # budget 20: 5 head + 15 tail
    assert short.split() == WORDS[:5] + WORDS[-15:]


@pytest.mark.parametrize("strategy", ["head", "head_tail"])
def test_encode_respects_max_length(tokenizer, strategy):
    ids = encode(tokenizer, [" ".join(WORDS)], max_length=22, truncation=strategy)["input_ids"][0]
    assert len(ids) == 22
    assert ids[0] == tokenizer.cls_token_id and ids[-1] == tokenizer.sep_token_id


def test_head_and_head_tail_differ_on_long_texts(tokenizer):
    text = " ".join(WORDS)
    head = encode(tokenizer, [text], 22, "head")["input_ids"][0]
    head_tail = encode(tokenizer, [text], 22, "head_tail")["input_ids"][0]
    assert tokenizer.decode(head_tail, skip_special_tokens=True).split()[-1] == "w99"
    assert head != head_tail


def test_encode_rejects_unknown_strategy(tokenizer):
    with pytest.raises(ValueError):
        encode(tokenizer, ["w1"], 22, "tail")
