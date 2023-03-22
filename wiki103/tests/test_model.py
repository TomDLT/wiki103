import torch

from wiki103.loader import Tokenizer, read_dataset
from wiki103.model import Transformer
from wiki103.trainer import Trainer

####################################################################################################
# load tokenizer
try:
    tokenizer = Tokenizer.load()
except FileNotFoundError:
    # creating the vocabulary takes about 2 minutes on Colab (102_587_097 tokens)
    word_generator = read_dataset(n_tokens=1, split="train")

    tokenizer = Tokenizer()
    tokenizer.fit(word_generator)
    tokenizer.save()

####################################################################################################


def test_tied_weights():
    # Test that the weights of the embedding layer and the softmax layer are tied

    model = Transformer(n_classes=tokenizer.n_classes_, cutoffs=[10000, 30000], n_blocks=2,
                        n_heads=4, n_tokens=256, n_embeddings=512)
    trainer = Trainer(model=model, tokenizer=tokenizer, learning_rate=0.001, n_epochs=1,
                      n_tokens=16, batch_size=2, weight_decay=0.01, accumulate_gradients=1)
    model = model.to("cuda")
    trainer.train(debug=True)

    assert torch.all(model.softmax.head.weight[:-2] == model.embedding.embeddings[0].weight).item()


def test_save_load():
    # Test that the model can be saved and loaded

    # train a model for a few steps
    model = Transformer(n_classes=tokenizer.n_classes_, cutoffs=[10000, 30000, 70000], n_blocks=2,
                        n_heads=4, n_tokens=256, n_embeddings=512)
    trainer = Trainer(model=model, tokenizer=tokenizer, learning_rate=0.001, n_epochs=1,
                      n_tokens=16, batch_size=2, weight_decay=0.01, accumulate_gradients=1)
    model = model.to("cuda")
    trainer.train(debug=True)

    # save the model
    model.save(custom_name="test_model")
    perplexity = trainer.compute_perplexity(debug=True)

    # reset the model, and load the saved model
    model.apply(model.init_weights)
    model.load(custom_name="test_model")
    model = model.to("cuda")
    perplexity2 = trainer.compute_perplexity(debug=True)

    assert perplexity == perplexity2
