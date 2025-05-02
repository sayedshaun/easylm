import langtrain as lt

def test_iterable_causal_dataset():
    tokenizer = lt.tokenizer.SentencePieceTokenizer(dir_or_path="tests/data", vocab_size=50, retrain=True)
    dataset = lt.data.IterableCausalDataset("tests/data", tokenizer, n_ctx=4)
    for x, y in dataset:
        assert x.shape == y.shape
        assert x.shape[0] == 4
        assert y.shape[0] == 4
        break
    assert dataset is not None
    assert dataset.tokenizer is not None


def test_lazy_causal_dataset():
    tokenizer = lt.tokenizer.SentencePieceTokenizer(dir_or_path="tests/data", vocab_size=50, retrain=True)
    dataset = lt.data.LazyCausalDataset("tests/data", tokenizer, n_ctx=4, token_caching=False)
    for x, y in dataset:
        print(x.shape, y.shape)
        assert x.shape == y.shape
        assert x.shape[0] == 4
        assert y.shape[0] == 4
        break
    assert dataset is not None
    assert dataset.tokenizer is not None
    assert dataset.n_ctx == 4
    assert len(dataset) > 0
    assert dataset.tokenizer.vocab_size == 50