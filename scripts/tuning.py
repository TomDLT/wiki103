from copy import deepcopy

import torch  # noqa
import numpy as np  # noqa
import matplotlib.pyplot as plt

from wiki103.loader import Tokenizer, read_dataset
from wiki103.model import Transformer
from wiki103.trainer import Trainer
from wiki103.config import parse_args

torch.manual_seed(0)

####################################################################################################
# load config
config_basis = parse_args()

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


def run_one(config):
    model = Transformer(
        n_classes=tokenizer.n_classes_,
        cutoffs=config["cutoffs"],
        n_blocks=config["n_blocks"],
        n_heads=config["n_heads"],
        n_tokens=config["n_tokens"],
        n_embeddings=config["n_embeddings"],
        tie_embedding=config["tie_embedding"],
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        learning_rate=config["learning_rate"],
        n_epochs=config["n_epochs"],
        n_tokens=config["n_tokens"],
        batch_size=config["batch_size"],
        weight_decay=config["weight_decay"],
        accumulate_gradients=config["accumulate_gradients"],
        print_loss_every_n_batches=64,
    )
    # disable saving
    model.save = lambda *args, **kwargs: None
    trainer.save = lambda *args, **kwargs: None
    trainer.plot = lambda *args, **kwargs: None

    model = model.to("cuda")
    trainer.train(debug=64 * 16)

    return np.array(trainer.training_losses_)


studies = {
    # "tie_embedding": [True, False],
    # "accumulate_gradients": [8, 16, 32, 64],
    "learning_rate": [0.001, 0.0005, 0.0003, 0.0001, 0.00003],
    # "weight_decay": [0.1, 0.01, 0.001],
}

for name, grid in studies.items():
    config = deepcopy(config_basis)

    colors = plt.get_cmap("viridis", len(grid)).colors
    fig, ax = plt.subplots()
    for ii, value in enumerate(grid):
        print(name, value)
        config[name] = value
        try:
            results = run_one(config)
        except KeyboardInterrupt:
            break

        ax.plot(results[:, 0], results[:, 1], label=value, color=colors[ii])
    ax.set(title=f"Tuning {name}", xlabel="n_tokens_seen", ylabel="Loss")
    ax.legend()
    ax.set(yscale="log", xscale="log")
    plt.savefig(f"tuning_{name}.png")
    plt.close()
