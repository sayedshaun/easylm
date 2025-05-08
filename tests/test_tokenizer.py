from langtrain.tokenizer import SentencePieceTokenizer

def test_tokenizer_initialization():
    tokenizer = SentencePieceTokenizer(dir_or_path="tests/data", vocab_size=50, retrain=True)
    assert tokenizer.vocab_size == 50
    assert tokenizer.mask_token_id is not None

def test_tokenizer_encode_decode():
    tokenizer = SentencePieceTokenizer(dir_or_path="tests/data", vocab_size=50, retrain=True)
    text = "adventures"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert isinstance(encoded, list)
    assert decoded.strip() == text.strip()

def test_tokenizer_save_and_load():
    tokenizer = SentencePieceTokenizer(dir_or_path="tests/data", vocab_size=50, retrain=True)
    tokenizer.save("tests/pretrained_model")
    loaded_tokenizer = SentencePieceTokenizer.from_pretrained("tests/pretrained_model")
    assert loaded_tokenizer.mask_token_id == tokenizer.mask_token_id