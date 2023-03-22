from functools import partial

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
from unidecode import unidecode

# force progress bar to overwrite
tqdm = partial(tqdm, position=0, leave=True)

dataset = load_dataset("wikitext", "wikitext-103-v1")


def read_dataset(n_tokens=512, split='train', n_lines_batch=1000, jitter=0, n_groups=None):
    """A generator that goes through a text dataset and yields groups of tokens.

    > We partition the training data into blocks of 512 contiguous tokens
    > ignoring document boundaries. Respecting document boundaries may lead to
    > better results and we leave this to future work.
    > [Baevski and Auli, 2019]

    Parameters
    ----------
    n_tokens : int
        Number of token in each group.
    split : str
        "train", "valid", "test"
    n_lines_batch : int
        NUmber of lines to read at once from the dataset loader.
    jitter : int
        Skip a fixed number of tokens at the beginning, to jitter the groups.
    n_groups : int or None
        If not None, stop the generator after returning a number of groups.

    Yields
    ------
    tokens : list of int, shape (n_tokens, )
        Tokens in groups of n_tokens.
    """
    generator = dataset[split].iter(n_lines_batch)
    jitter = jitter if jitter not in [None, False] else 0

    count = 0
    current = ""
    for batch in generator:
        # merge the batch and clean up
        merged = ''.join(batch['text']).strip()
        # change non-ascii into closest ascii
        merged = unidecode(merged)
        # remove uppercase letters
        merged = merged.lower()
        merged = merged.replace("@-@", "-")
        # remove double spaces
        while "  " in merged:
            merged = merged.replace("  ", " ")

        # add to current buffer
        current += " " + merged
        # split into tokens, yield group of tokens
        tokens = current.split(" ")
        if jitter > 0:  # skip a fixed number, to offset the blocks
            jitter, tokens = jitter - len(tokens[:jitter]), tokens[jitter:]
        stop = 0
        for start in range(0, len(tokens) - n_tokens, n_tokens):
            yield tokens[start:start + n_tokens]
            stop = start + n_tokens
            count += 1

        if n_groups is not None and count >= n_groups:
            break

        # leftover tokens
        current = " ".join(split[stop:])

    # ignore last incomplete group


class Tokenizer():
    """Tokenizer sorting by most frequent words first.

    Parameters
    ----------
    ignored_frequencies : int
        Ignore words that appear less than a number of times in the training
        set.
    """

    def __init__(self, ignored_frequencies=5):
        self.ignored_frequencies = ignored_frequencies

    def fit(self, word_generator):
        # create vocabulary from training data
        vocab = dict()
        for word in tqdm(word_generator):
            vocab[word[0]] = vocab.get(word[0], 0) + 1
        self.n_train_tokens_ = sum(freq for freq in vocab.values())

        # remove infrequent words, make a list
        vocab = [[key, val] for key, val in vocab.items() if val > self.ignored_frequencies]
        # sort by frequency, most frequent first
        vocab = sorted(vocab, key=lambda x: x[1])[::-1]

        # save attributes
        self.frequencies_ = np.array([freq for word, freq in vocab])
        self.encoder_ = {word: index for index, (word, freq) in enumerate(vocab)}
        self.decoder_ = {index: word for index, (word, freq) in enumerate(vocab)}
        self.n_classes_ = len(self.frequencies_)

        return self

    def encode(self, word_list):
        return [
            self.encoder_[word] if word in self.encoder_ else self.encoder_["<unk>"]
            for word in word_list
        ]

    def decode(self, index_list):
        return [self.decoder_[index] for index in index_list]

    def plot(self):
        frequencies = self.frequencies_
        ax = plt.gca()
        bins = np.logspace(np.log10(frequencies.min()), np.log10(frequencies.max()), 101)
        ax.hist(frequencies, bins, log=True, alpha=0.7)
        ax.set(xlabel='word frequency', ylabel="number of words", xscale="log")
        plt.show()

    def save(self, filename="vocabulary.npy"):
        np.save(filename, np.array([key for key in self.encoder_.keys()]))

    @classmethod
    def load(cls, filename="vocabulary.npy"):
        vocab = np.load(filename)
        new = cls()
        new.encoder_ = {word: index for index, word in enumerate(vocab)}
        new.decoder_ = {index: word for index, word in enumerate(vocab)}
        new.n_classes_ = len(vocab)
        new.frequencies_ = np.ones(len(vocab))
        return new


def make_batch_generator(block_generator, tokenizer, batch_size=2):
    """Consume generator of blocks of token, apply tokenizer, make batches"""
    batch = []
    for block in tqdm(block_generator):
        encoded = tokenizer.encode(block)
        batch.append(encoded)
        if len(batch) < batch_size:
            continue
        else:
            yield torch.as_tensor(batch)
            batch = []
