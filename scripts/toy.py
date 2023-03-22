import torch  # noqa
from wiki103.loader import Tokenizer, read_dataset
from wiki103.model import Transformer
from wiki103.trainer import Trainer
from wiki103.config import parse_args

####################################################################################################
# load config
config = parse_args()

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
# model definition (smaller, just for toy testing)

model = Transformer(
    n_classes=tokenizer.n_classes_,
    cutoffs=config["cutoffs"],
    n_blocks=4,
    n_heads=4,
    n_tokens=config["n_tokens"],
    n_embeddings=128,
)

model.count_parameters()

####################################################################################################
# train the model


class ToyTrainer(Trainer):
    """Toy trainer, looping over a single batch, to check overfitting."""
    def reset_training_generator(self):
        from wiki103.loader import make_batch_generator
        block_generator = read_dataset(n_tokens=self.n_tokens + 1, split="train")
        self.batch_size = 1
        self.batch_generator_ = make_batch_generator(block_generator, self.tokenizer,
                                                     batch_size=self.batch_size)
        self._batch = next(self.batch_generator_)

        def batch_generator_():
            for ii in range(0, self.n_tokens):
                yield torch.roll(self._batch, ii, dims=1)

        self.batch_generator_ = batch_generator_()

    def compute_validation(self):
        prompt = "commonly referred to as"
        trainer.generate_text(n_tokens=2, prompt=prompt, temperature=0, do_print=True)


trainer = ToyTrainer(
    model=model,
    tokenizer=tokenizer,
    learning_rate=config["learning_rate"],
    n_epochs=config["n_epochs"],
    n_tokens=config["n_tokens"],
    batch_size=config["batch_size"],
    weight_decay=config["weight_decay"],
    accumulate_gradients=config["accumulate_gradients"],
)

if not config["no-train"]:
    if not config["from-scratch"]:
        trainer.load()
    trainer.save()
    try:
        model = model.to("cuda")
        trainer.train()
    except KeyboardInterrupt:
        model.save()
        trainer.save()
        pass
else:
    print("No training.")
    trainer.load(fast_forward=False)

####################################################################################################
# example of text generation

full_prompt = " ".join(trainer.tokenizer.decode(trainer._batch[0, :50].cpu().numpy()))
print(full_prompt)
for ii in range(20):
    prompt = " ".join(full_prompt.split(" ")[:ii])
    trainer.generate_text(n_tokens=2, prompt=prompt, temperature=0, do_print=True)

import IPython  # noqa 
IPython.embed(using=False)
