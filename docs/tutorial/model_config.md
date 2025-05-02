## ⚙️ Model Hyperparameters

| Parameter     | Type              | Default | Description                                          |
|---------------|-------------------|---------|------------------------------------------------------|
| `vocab_size`  | `int` or `None`   | `None`  | Size of the vocabulary (number of tokens).           |
| `hidden_size` | `int` or `None`   | `None`  | Dimensionality of the hidden states.                 |
| `num_heads`   | `int` or `None`   | `None`  | Number of attention heads.                           |
| `num_layers`  | `int` or `None`   | `None`  | Number of transformer layers.                        |
| `norm_epsilon`| `float` or `None` | `None`  | Epsilon value for layer normalization stability.     |
| `dropout`     | `float` or `None` | `None`  | Dropout probability for regularization.              |
| `max_seq_len` | `int` or `None`   | `None`  | Maximum sequence length the model can handle.        |