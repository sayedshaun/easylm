import langtrain as lt

def test_llama_trainer():
    tokenizer = lt.tokenizer.SentencePieceTokenizer(dir_or_path="tests/data", vocab_size=50, retrain=True)
    dataset = lt.data.SimpleCausalDataset("tests/data", tokenizer, n_ctx=4)
    model = lt.model.LlamaModel(
        config=lt.config.LlamaConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=16,
            num_heads=1,
            num_layers=1,
            dropout=0.1,
            max_seq_len=4,
            norm_epsilon=1e-5
        )
    )
    train_config = lt.config.TrainingConfig(
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        logging_steps=1,
        save_steps=1,
        device="cpu",
        monitor_loss_for="val",
        early_stopping=True,
        patience=1,
        overwrite_output_dir=True
    )
    trainer = lt.trainer.Trainer(
        model=model, 
        tokenizer=tokenizer,
        train_data=dataset,
        val_data=dataset,
        test_data=dataset,
        model_name="tests/test_weights", 
        config=train_config)
    assert trainer is not None
    assert trainer.model is not None
    assert trainer.tokenizer is not None
    trainer.train()
    model.from_pretrained("tests/test_weights")
    trainer.predict(dataset, train_config.device)

def test_gpt_trainer():
    tokenizer = lt.tokenizer.SentencePieceTokenizer(dir_or_path="tests/data", vocab_size=50, retrain=True)
    dataset = lt.data.SimpleCausalDataset("tests/data", tokenizer, n_ctx=4)
    model = lt.model.GPTModel(
        config=lt.config.GPTConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=16,
            num_heads=1,
            num_layers=1,
            dropout=0.1,
            max_seq_len=4,
            norm_epsilon=1e-5
        )
    )
    train_config = lt.config.TrainingConfig(
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        logging_steps=1,
        save_steps=1,
        device="cpu",
        monitor_loss_for="val",
        overwrite_output_dir=True
    )
    trainer = lt.trainer.Trainer(
        model=model, 
        tokenizer=tokenizer,
        train_data=dataset,
        val_data=dataset,
        test_data=dataset,
        model_name="tests/test_weights", 
        config=train_config)
    assert trainer is not None
    assert trainer.model is not None
    assert trainer.tokenizer is not None
    trainer.train()
    model.from_pretrained("tests/test_weights")
    trainer.predict(dataset, train_config.device)


def test_bert_trainer():
    tokenizer = lt.tokenizer.SentencePieceTokenizer(dir_or_path="tests/data", vocab_size=50, retrain=True)
    dataset = lt.data.SimpleCausalDataset("tests/data", tokenizer, n_ctx=4)
    model = lt.model.BertModel(
        config=lt.config.BertConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=16,
            num_heads=1,
            num_layers=1,
            dropout=0.1,
            max_seq_len=4,
            norm_epsilon=1e-5
        )
    )
    train_config = lt.config.TrainingConfig(
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        logging_steps=1,
        save_steps=1,
        device="cpu",
        early_stopping=5,
        patience=2,
        overwrite_output_dir=True,
        monitor_loss_for="val",
    )
    trainer = lt.trainer.Trainer(
        model=model, 
        tokenizer=tokenizer,
        train_data=dataset,
        # val_data=dataset,
        test_data=dataset,
        model_name="tests/test_weights", 
        config=train_config)
    assert trainer is not None
    assert trainer.model is not None
    assert trainer.tokenizer is not None
    trainer.train()
    model.from_pretrained("tests/test_weights")
    trainer.predict(dataset, train_config.device)



