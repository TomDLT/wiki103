# Adaptive-embedding transformer trained on wikitext-103-v1

The goal is to replicate the model described in
[Adaptive input representations for neural language modeling](https://arxiv.org/abs/1809.10853) [Baevski and Auli, 2019], and train it on `wikitext-103-v1`.

## TODO

- [x] implement dataset loader and tokenizer
- [x] implement adaptive embedding
- [x] implement decoder-only transformer based on `torch.nn.MultiheadAttention`
- [x] use adaptive softmax from `torch.nn.AdaptiveLogSoftmaxWithLoss`
- [x] match adaptive embedding API with adaptive softmax
- [x] tie the embedding and softmax weights
- [x] implement training loop, loss monitoring, learning rate schedule
- [x] implement model save/load, continue training where it stopped
- [x] aggregate the gradient over multiple batches
- [x] check underfitting by experimenting with a tiny training set
- [x] implement a function to compute perplexity on the validation set
- [x] fix save/load model, fix weight tying
- [x] improve the adaptive softmax to be able to use float16 mixed-precision
- [ ] try full size (n_tokens=3072)

##  Model fitting

```bash
python run.py --big
```

- Large model: n_blocks=16, n_heads=16, n_tokens=1024, n_embeddings=1024
- Adaptive embedding: cutoffs=[20000, 60000]

![transformer_16x16x1024x1024x20000x60000 pt_losses](https://user-images.githubusercontent.com/11065596/225403928-fbf2ff80-475a-4ad1-a6c4-81a3e25d5912.png)

