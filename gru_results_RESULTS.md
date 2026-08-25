# GRU vs Transformer — Results

Config: `{"csv_path": "data/IMDB-Dataset.csv", "max_samples": 6000, "max_len": 120, "vocab_size": 20000, "batch_size": 32, "epochs": 4, "embed_dim": 100, "hidden_dim": 100, "num_layers": 2, "num_heads": 4, "lr": 0.001, "out_dir": "results"}`

| Model | Params | Train time (s) | Test Acc | Test F1 | Test AUC | Latency (ms/ex) |
|---|---|---|---|---|---|---|
| GRU | 2,302,801 | 280.1 | 0.7067 | 0.7067 | 0.7838 | 1.166 |
| Transformer | 2,162,501 | 192.8 | 0.7311 | 0.7623 | 0.8104 | 1.512 |