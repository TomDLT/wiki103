import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler

from .loader import read_dataset, make_batch_generator


class Trainer():
    def __init__(self, model, tokenizer, learning_rate, n_epochs=10, n_tokens=512, batch_size=10,
                 weight_decay=0.01, accumulate_gradients=1, print_loss_every_n_batches=128,
                 validation_every_n_batches=2048, mixed_precision=False):
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_tokens = n_tokens
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.accumulate_gradients = accumulate_gradients
        self.print_loss_every_n_batches = print_loss_every_n_batches
        self.validation_every_n_batches = validation_every_n_batches
        self.mixed_precision = mixed_precision

        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                     weight_decay=weight_decay)
        self.n_tokens_train = 102_587_097

        self.n_tokens_seen_ = 0
        self.training_losses_ = []
        self.validation_losses_ = []
        self.validation_perplexities_ = []
        self.reset_training_generator()

    def reset_training_generator(self, fast_forward=False):
        """Reset the training generator. This is done at the beginning of each epoch.

        Parameters
        ----------
        fast_forward : bool
            If True, fast forward to the point where we stopped in the previous epoch.
        """
        # add jitter in subsequent epochs
        if self.n_epochs_seen > 0.99:
            jitter = np.random.randint(self.n_tokens)
            n_tokens = self.n_tokens
            batch_size = self.batch_size
        else:
            jitter = 0
            n_tokens = self.n_tokens  # // 4  # use smaller blocks in the first epoch
            batch_size = self.batch_size  # * 4

        # create a new generator
        block_generator = read_dataset(n_tokens=n_tokens + 1, split="train", jitter=jitter)
        self.batch_generator_ = make_batch_generator(block_generator, self.tokenizer,
                                                     batch_size=batch_size)

        # skip to where we stopped
        if fast_forward:
            print("Fast forwarding to i_tokens = %d" % (self.n_tokens_seen_ % self.n_tokens_train))
            n_tokens_seen = 0
            while n_tokens_seen < self.n_tokens_seen_ % self.n_tokens_train:
                next(self.batch_generator_)
                n_tokens_seen += n_tokens * batch_size

    def train(self, debug=False):
        """Train the model for n_epochs."""

        if self.mixed_precision:
            scaler = GradScaler()
        while self.n_epochs_seen < self.n_epochs:
            self.optimizer.zero_grad()
            accum_loss = 0
            for ii, batch in enumerate(self.batch_generator_):

                # forward/backward pass
                batch = batch.to(self.model.device, non_blocking=True)
                input, target = batch[:, :-1], batch[:, 1:]

                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        _, this_loss = self.model.forward(input, target)
                    scaler.scale(this_loss).backward()
                else:
                    _, this_loss = self.model.forward(input, target)
                    this_loss.backward()
                assert not np.isnan(this_loss.item())

                # gradient step
                if (ii + 1) % self.accumulate_gradients == 0:
                    self.update_learning_rate()
                    if self.mixed_precision:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       5.0 * self.accumulate_gradients)
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       5.0 * self.accumulate_gradients)
                        self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True)

                # accumulate loss for logging
                accum_loss += this_loss.item()
                self.n_tokens_seen_ += input.numel()
                if (ii + 1) % self.print_loss_every_n_batches == 0:
                    self.log_training_loss(accum_loss / self.print_loss_every_n_batches)
                    accum_loss = 0
                if (ii + 1) % self.validation_every_n_batches == 0:
                    self.model.save()
                    self.compute_validation()
                    self.save()
                    self.plot()

                if debug and ii > debug:
                    break
            if debug:
                break

            # end of epoch
            self.compute_validation()
            self.reset_training_generator()

    @property
    def n_epochs_seen(self):
        return self.n_tokens_seen_ / self.n_tokens_train

    def log_training_loss(self, average_loss):
        print(f'\r Epoch {np.floor(self.n_epochs_seen):.0f}/{self.n_epochs}, '
              f'Token {self.n_tokens_seen_ / 1e6:.1f}'
              f'/{self.n_tokens_train /1e6:.0f}M, '
              f'Loss: {average_loss:.4f}')
        self.training_losses_.append([self.n_tokens_seen_, average_loss])

    def compute_validation(self):
        block_generator = read_dataset(n_tokens=self.n_tokens + 1, split="validation", jitter=0)
        batch_generator = make_batch_generator(block_generator, self.tokenizer,
                                               batch_size=self.batch_size)
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            n_batches = 0
            for batch in batch_generator:
                batch = batch.to(self.model.device, non_blocking=True)
                input, target = batch[:, :-1], batch[:, 1:]
                _, this_loss = self.model.forward(input, target)
                total_loss += this_loss.item()
                n_batches += 1

        average = total_loss / n_batches
        print(f'Validation Loss: {average:.4f}')
        self.validation_losses_.append([self.n_tokens_seen_, average])

        self.compute_perplexity()

    def compute_perplexity(self, stride=0.25, overlap=False, debug=False):
        """Compute perplexity on validation set.

        Parameters
        ----------
        stride : int or float
            Number of tokens to compute perplexity on. The rest of the tokens are only used
            for context. If float, it is interpreted as a fraction of n_tokens.
        overlap : bool
            If True, compute perplexity on all tokens in the validation set, using a sliding
            window and an overlap of size `stride`. If False, the windows are not overlapping.
        debug : bool
            If True, only compute perplexity on the first block of n_tokens.
        """
        from torch.nn import functional as F

        if isinstance(stride, float):
            stride = int(stride * self.n_tokens)
            assert stride > 0

        jitter_range = range(0, self.n_tokens, stride) if overlap and not debug else [0]

        self.model.eval()
        with torch.no_grad():
            total_neg_log_like = 0
            n_blocks = 0
            for jitter in jitter_range:
                block_generator = read_dataset(n_tokens=self.n_tokens + 1, split="validation",
                                               jitter=jitter)
                for block in block_generator:
                    encoded = self.tokenizer.encode(block)
                    encoded = torch.as_tensor(encoded).to(self.model.device, non_blocking=True)
                    input, target = encoded[:-1], encoded[1:]
                    log_prob = self.model.log_prob(input)
                    neg_log_like = F.cross_entropy(log_prob[-stride:], target[-stride:])
                    total_neg_log_like += neg_log_like.item()
                    n_blocks += 1

                    if debug:
                        break

        perplexity = np.exp(total_neg_log_like / n_blocks)
        print(f'Perplexity: {perplexity:.4f}')
        self.validation_perplexities_.append([self.n_tokens_seen_, perplexity])
        return perplexity

    def save_name(self):
        return f'{self.model.save_name()}_losses.npz'

    def save(self):
        np.savez(
            self.save_name(),
            training_loss=np.array(self.training_losses_),
            validation_loss=np.array(self.validation_losses_),
            validation_perplexity=np.array(self.validation_perplexities_),
            n_tokens_seen=self.n_tokens_seen_,
        )

    def load(self, fast_forward=True):
        try:
            data = np.load(self.save_name())
            self.training_losses_ = data['training_loss'].tolist()
            self.validation_losses_ = data['validation_loss'].tolist()
            self.validation_perplexities_ = data['validation_perplexity'].tolist()
            self.n_tokens_seen_ = int(data['n_tokens_seen'])
            self.reset_training_generator(fast_forward=fast_forward)
            print("trainer loaded")
        except Exception:
            print("trainer not loaded")

    def learning_rate_scheduler(self, n_tokens_seen, learning_rate, warmup=2e8, decay=8e8,
                                rescale=0.75):
        """Learning rate scheduler."""

        if n_tokens_seen < warmup:
            # linear warmup from 0 to `learning_rate`
            return learning_rate * (n_tokens_seen + 1) / warmup
        else:
            n_tokens_seen = n_tokens_seen - warmup
            
        # every `decay` tokens, we rescale the learning rate by `rescale`
        n_cycles = int(n_tokens_seen / decay)
        n_tokens_seen = n_tokens_seen - n_cycles * decay
        learning_rate = learning_rate * rescale ** n_cycles

        # linear decay from `learning_rate` to `learning_rate / 10`
        min_learning_rate = learning_rate / 10
        ramp = max(0, (decay - n_tokens_seen) / decay)
        return min_learning_rate + ramp * (learning_rate - min_learning_rate)

    def update_learning_rate(self):
        learning_rate = self.learning_rate_scheduler(self.n_tokens_seen_, self.learning_rate)
        assert 0 < learning_rate <= self.learning_rate, str(learning_rate)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

    def plot(self):
        import numpy as np
        import matplotlib.pyplot as plt

        training_losses_ = np.array(self.training_losses_)
        validation_losses_ = np.array(self.validation_losses_)
        validation_perplexities_ = np.array(self.validation_perplexities_)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, slice_ in zip(axes, [
                slice(0, None),
                slice(-len(training_losses_) // 4, None),
        ]):
            ax.plot(training_losses_[slice_, 0], training_losses_[slice_, 1], label='training',
                    alpha=0.7)
            if len(validation_losses_) > 0:
                mask = validation_losses_[:, 0] >= training_losses_[slice_, 0].min()
                ax.loglog(validation_losses_[mask, 0], validation_losses_[mask, 1],
                          label='validation', marker=".")
            ax.set_xlabel('n_tokens_seen')
            ax.set_ylabel('loss')
            ax.legend()

            if slice_.start != 0:
                ax.set(xscale="linear", yscale="linear")
                ymax = max(training_losses_[slice_, 1][:1].max(),
                           validation_losses_[mask, 1][:1].max())
                ymin = min(training_losses_[slice_, 1].min(), validation_losses_[mask, 1].min())
                ax.set_ylim(ymin - (ymax - ymin) * 0.05, ymax + (ymax - ymin) * 0.05)

            if len(validation_perplexities_) > 0:
                color = "C2"
                mask = validation_perplexities_[:, 0] >= training_losses_[slice_, 0].min()

                ax2 = ax.twinx()
                ax2.set_ylabel('perplexity', color=color)
                ax2.tick_params(axis='y', labelcolor=color)
                ax2.loglog(validation_perplexities_[mask, 0], validation_perplexities_[mask, 1],
                           label='validation', marker=".", color=color)
                if slice_.start != 0:
                    ax2.set(xscale="linear", yscale="linear")

                    ymax = validation_perplexities_[mask, 1][:1].max()
                    ymin = validation_perplexities_[mask, 1].min()
                    ax2.set_ylim(ymin - (ymax - ymin) * 0.05, ymax + (ymax - ymin) * 0.05)

        fig.tight_layout()
        fig.savefig(f'{self.model.save_name()}_losses.png')
        plt.show()
        plt.close(fig)

    def generate_text(self, prompt="", n_tokens=10, temperature=0, top_p=0.999, do_print=False):
        """Generate text with the model.

        Parameters
        ----------
        prompt : str
            Text to use as a prompt.
        n_tokens : int
            Number of tokens to generate.
        temperature : float
            Temperature for the softmax. If 0, the argmax is used.
        top_p : float
            Considered only the top tokens for the softmax, up to cumulated probability p.
            If 1, all tokens are considered.
        do_print : bool
            Whether to print the generated text using colorify.
        """
        from unidecode import unidecode
        # Clean up the seed text
        prompt = unidecode(prompt)
        prompt = prompt.lower().strip()
        prompt = prompt.replace("@-@", "-")
        while "  " in prompt:
            prompt = prompt.replace("  ", " ")
        words = prompt.split(" ")

        # Generate text with the model
        self.model.eval()
        with torch.no_grad():
            input_tokens = self.tokenizer.encode(words)
            input_tokens = torch.tensor(input_tokens)
            input_tokens = input_tokens.to(self.model.device, non_blocking=True)
            output_tokens = []
            tokens = input_tokens[-self.n_tokens:]
            for _ in range(n_tokens):
                next_tokens = self.model.predict(tokens, temperature=temperature, top_p=top_p)
                output_tokens.append(next_tokens[-1].item())
                tokens = torch.cat([tokens[-(self.n_tokens - 1):], next_tokens[-1:]], dim=0)

        output_words = self.tokenizer.decode(output_tokens)
        output_text = " ".join(output_words)

        if do_print:

            def colorify(message, color=34):
                return ("\033[1;%dm" % color) + message + "\033[0m"

            print(prompt + " " + colorify(output_text))

        return output_text
