import torch  # noqa
from wiki103.loader import Tokenizer, read_dataset
from wiki103.model import Transformer
from wiki103.trainer import Trainer
from wiki103.config import parse_args

torch.manual_seed(0)

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
    # tokenizer.plot()
    tokenizer.save()

####################################################################################################
# model definition
model = Transformer(
    n_classes=tokenizer.n_classes_,
    cutoffs=config["cutoffs"],
    n_blocks=config["n_blocks"],
    n_heads=config["n_heads"],
    n_tokens=config["n_tokens"],
    n_embeddings=config["n_embeddings"],
    tie_embedding=config["tie_embedding"],
)

if config["from-scratch"]:
    print("model retrained from scratch")
else:
    model.load()

model.count_parameters()

####################################################################################################
# train the model
trainer = Trainer(
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
    else:
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
    trainer.plot()

####################################################################################################
# example of text generation

prompt = "The quick brown fox jumps over the"
text = trainer.generate_text(prompt=prompt, n_tokens=10, temperature=0, do_print=True)

####################################################################################################
import IPython  # noqa
IPython.embed(using=False)
