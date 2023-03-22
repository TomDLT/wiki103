import os
import socket

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from .softmax import AdaptiveLogSoftmaxWithLoss

if socket.gethostname() == "nyquil":
    save_path = "/data1/models/wiki103/"
else:
    save_path = "/home/jlg/tomdlt/models/wiki103/"


class AdaptiveEmbedding(nn.Module):
    """Embedding with a different number of dimensions per word clusters,
    defining clusters based on similar word frequencies.

    Parameters
    ----------
    n_classes : int
        Number of classes in the vocabulary.
    cutoffs : list of int
        Cutoffs defining the clusters.
    n_embeddings : int
        Number of dimensions for the embeddings.
    reduction_factor : int
        Factor by which the number of dimensions is reduced for each cluster.
    """

    def __init__(self, n_classes, cutoffs, n_embeddings, reduction_factor=4):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.reduction_factor = reduction_factor

        # use fixed sizes for the clusters
        self.n_classes = n_classes
        self.n_clusters = len(cutoffs) + 1
        self.cutoffs = cutoffs
        self.cutoffs_extended = [0] + list(cutoffs) + [n_classes]

        # create embedding for each cluster
        # and linear mapping to get same number of dimension at the end
        self.embeddings = nn.ModuleList()
        self.mappings = nn.ModuleList()
        n_embeddings_ii = self.n_embeddings
        for ii in range(self.n_clusters):
            cluster_count = self.cutoffs_extended[ii + 1] - self.cutoffs_extended[ii]
            embedding = nn.Embedding(cluster_count, n_embeddings_ii)
            mapping = nn.Linear(n_embeddings_ii, self.n_embeddings, bias=False)
            self.embeddings.append(embedding)
            self.mappings.append(mapping)
            n_embeddings_ii = n_embeddings_ii // reduction_factor

    def forward(self, indices):
        out = None
        for ii in range(self.n_clusters):
            mask = torch.logical_and(self.cutoffs_extended[ii] <= indices,
                                     indices < self.cutoffs_extended[ii + 1])
            relative_indices = indices[mask] - self.cutoffs_extended[ii]

            xx = self.embeddings[ii](relative_indices)
            xx = self.mappings[ii](xx)
            if out is None:
                out = indices.new_empty(list(indices.shape) + [self.n_embeddings], dtype=xx.dtype)
            out[mask, :] = xx

        return out


class PositionalEmbedding(nn.Module):
    """Positional embedding as described in Attention is all you need.

    Parameters
    ----------
    n_embeddings : int
        Number of dimensions for the embeddings.
    n_tokens : int
        Number of tokens in the sequence.
    """

    def __init__(self, n_embeddings=1024, n_tokens=512):
        super().__init__()
        theta = np.array(
            [[position / np.power(10000, 2 * ii / n_embeddings) for position in range(n_tokens)]
             for ii in range(n_embeddings)]).T
        theta[:, 0::2] = np.sin(theta[:, 0::2])
        theta[:, 1::2] = np.cos(theta[:, 1::2])
        theta = torch.Tensor(theta)
        self.embedding = nn.Embedding.from_pretrained(theta, freeze=True)

    def forward(self, xx):
        position = torch.arange(xx.shape[-1], dtype=torch.long, device=xx.device)
        if self.embedding.weight.device != xx.device:
            self.embedding = self.embedding.to(xx.device)
        xx = self.embedding(position)
        return xx

    def plot(self):
        ax = plt.gca()
        ax.imshow(self.embedding.weight.cpu().numpy())
        ax.set(xlabel="embedding dimension", ylabel="token position")
        plt.show()


class MultiHeadAttentionBlock(nn.Module):
    """Multi-head attention block with causal mask.

    Parameters
    ----------
    n_embeddings : int
        Number of dimensions for the embeddings.
    n_heads : int
        Number of heads for the multi-head attention.
    dropout : float
        Dropout rate.
    """

    def __init__(self, n_embeddings, n_heads, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_embeddings)
        self.attention = nn.MultiheadAttention(n_embeddings, n_heads, dropout=dropout,
                                               batch_first=True)

    def forward(self, xx):
        xx = self.layer_norm(xx)
        attn_shape = (xx.shape[-2], xx.shape[-2])
        attn_mask = ~torch.tril(torch.ones(attn_shape, dtype=bool, device=xx.device))
        xx, _ = self.attention(xx, xx, xx, attn_mask=attn_mask)
        return xx


class FeedForwardBlock(nn.Module):
    """Feed-forward block with dropout.

    Parameters
    ----------
    n_embeddings : int
        Number of dimensions for the embeddings.
    n_expand : int
        Number of dimensions for the intermediate layer.
    dropout : float
        Dropout rate.
    """

    def __init__(self, n_embeddings, n_expand, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_embeddings)
        self.linear_0 = nn.Linear(n_embeddings, n_expand, bias=True)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(n_expand, n_embeddings, bias=True)

    def forward(self, xx):
        xx = self.layer_norm(xx)
        xx = self.linear_0(xx)
        xx = self.relu(xx)
        xx = self.linear_1(xx)
        xx = self.dropout(xx)
        return xx


class Block(nn.Module):
    """Transformer block with multi-head attention and feed-forward.

    Parameters
    ----------
    n_embeddings : int
        Number of dimensions for the embeddings.
    n_heads : int
        Number of heads for the multi-head attention.
    dropout_attention : float
        Dropout rate for the multi-head attention.
    dropout_feed_forward : float
        Dropout rate for the feed-forward.
    """

    def __init__(self, n_embeddings, n_heads=16, dropout_attention=0.1, dropout_feed_forward=0.1):
        super().__init__()
        self.attention = MultiHeadAttentionBlock(n_embeddings, n_heads, dropout_attention)
        self.feed_forward = FeedForwardBlock(n_embeddings, 4 * n_embeddings, dropout_feed_forward)

    def forward(self, xx):
        xx = xx + self.attention(xx)
        xx = xx + self.feed_forward(xx)
        return xx


###############################################################################


class Transformer(nn.Module):
    """Transformer model for language modeling.

    Parameters
    ----------
    n_classes : int
        Number of classes in the vocabulary.
    cutoffs : list of int
        Cutoffs for the adaptive embeddings.
    n_blocks : int
        Number of transformer blocks.
    n_heads : int
        Number of heads for the multi-head attention.
    n_tokens : int
        Number of tokens in the sequence.
    n_embeddings : int
        Number of dimensions for the embeddings.
    tie_embedding : bool
        Whether to tie the embedding and softmax output layers.
    """

    def __init__(self, n_classes, cutoffs=[20000, 60000], n_blocks=16, n_heads=16, n_tokens=512,
                 n_embeddings=1024, tie_embedding=True):
        super().__init__()
        self.n_classes = n_classes
        self.cutoffs = cutoffs
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.n_tokens = n_tokens
        self.n_embeddings = n_embeddings
        self.tie_embedding = tie_embedding

        # layers
        self.embedding = AdaptiveEmbedding(n_classes=n_classes, cutoffs=cutoffs,
                                           n_embeddings=n_embeddings)
        self.positional = PositionalEmbedding(n_embeddings=n_embeddings, n_tokens=n_tokens)
        self.blocks = nn.ModuleList(
            [Block(n_embeddings=n_embeddings, n_heads=n_heads) for _ in range(n_blocks)])

        # the last linear layer is included in the adaptive softmax
        self.softmax = AdaptiveLogSoftmaxWithLoss(in_features=n_embeddings, n_classes=n_classes,
                                                  cutoffs=cutoffs)

        # init the weights with Gaussian noise
        self.apply(self.init_weights)

        # weight tying between embedding and softmax.
        self.tie_all_weights()

    def tie_all_weights(self):
        """Tie the weights of the adaptive softmax and the adaptive embedding.

        WARNING: Must be called after model.to(device), otherwise the weights will not be shared.
        # related to https://github.com/pytorch/xla/issues/2719
        TODO: use a more robust method to tie the weights.
        """
        if self.tie_embedding:
            self._tie_weights(self.embedding.embeddings[0], self.softmax.head,
                              index_stop_b=-len(self.cutoffs))
            for ii in range(len(self.embedding.embeddings) - 1):
                self._tie_weights(self.embedding.embeddings[ii + 1], self.softmax.tail[ii][1])
                self._tie_weights(self.embedding.mappings[ii + 1], self.softmax.tail[ii][0],
                                  transpose=True)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.tie_all_weights()
        return self

    def _tie_weights(self, layer_a, layer_b, transpose=False, index_stop_b=None):
        """Tie the weights of two layers."""
        if transpose:
            assert (layer_a.weight.shape == layer_b.weight[:index_stop_b].T.shape)
            layer_a.weight = nn.Parameter(layer_b.weight[:index_stop_b].transpose(0, 1))
        else:
            assert layer_a.weight.shape == layer_b.weight[:index_stop_b].shape
            layer_a.weight = nn.Parameter(layer_b.weight[:index_stop_b])

    def _forward_no_softmax(self, xx):
        xx = self.embedding(xx) + self.positional(xx)
        for block in self.blocks:
            xx = block(xx)
        return xx

    def forward(self, xx, target):
        """Forward pass through the model, used during training."""
        xx = self._forward_no_softmax(xx)

        if target.ndim == 2:  # adaptive softmax does not deal with batches
            outputs, loss = self.softmax(xx.view(-1, self.n_embeddings), target.reshape(-1))
            outputs = outputs.view(*target.shape)
        else:
            outputs, loss = self.softmax(xx, target)
        return outputs, loss

    def log_prob(self, xx):
        """Forward pass through the model, used for prediction."""
        assert xx.ndim == 1, "Adaptive softmax does not deal with batches."
        xx = self._forward_no_softmax(xx)
        log_prob = self.softmax.log_prob(xx)
        return log_prob

    def predict(self, xx, temperature=0, top_p=0.99):
        """Forward pass through the model, used for prediction."""
        assert xx.ndim == 1, "Adaptive softmax does not deal with batches."
        xx = self._forward_no_softmax(xx)

        if temperature == 0:
            predictions = self.softmax.predict(xx)
            return predictions

        log_prob = self.softmax.log_prob(xx)
        log_prob = log_prob.div(temperature)

        if top_p is None or top_p == 1:
            probs = F.softmax(log_prob.exp(), dim=-1)
            predictions = torch.multinomial(probs, 1).squeeze(-1)
        else:
            # top-p sampling
            sorted_log_prob, sorted_indices = torch.sort(log_prob, descending=True)
            sorted_log_prob -= sorted_log_prob[:, :1].clone()
            cumulative_prob = torch.cumsum(sorted_log_prob.exp(), dim=-1)
            cumulative_prob /= cumulative_prob[:, -1:].clone()
            sorted_indices_to_remove = cumulative_prob > top_p
            sorted_indices_to_remove[:, 0] = 0  # always keep the first token
            probs = sorted_log_prob.exp()
            probs[sorted_indices_to_remove] = 0
            probs /= probs.sum(dim=-1, keepdim=True)
            predictions = torch.multinomial(probs, 1).squeeze(-1)
            predictions = sorted_indices.gather(1, predictions.unsqueeze(-1))
            predictions = predictions.squeeze(-1)

        return predictions

    def init_weights(self, module):
        """Initialize the weights of the model with Gaussian distributions."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.01)

    def save_name(self, custom_name=None):
        """Return the name of the file where the model is saved."""
        if custom_name is None:
            name = "transformer_"
            for param in [
                    self.n_blocks,
                    self.n_heads,
                    self.n_tokens,
                    self.n_embeddings,
                    *self.cutoffs,
            ]:
                name += f"{param}x"
            name = name[:-1] + ".pt"
        else:
            name = custom_name
        name = os.path.join(save_path, name)
        return name

    def save(self, custom_name=None):
        """Save the model on disk."""
        torch.save(self.state_dict(), self.save_name(custom_name))
        print("model saved")

    def load(self, custom_name=None):
        """Load the model if it exists."""
        try:
            self.load_state_dict(torch.load(self.save_name(custom_name)))
            self.tie_all_weights()
            print("model loaded")
        except Exception:
            print("model not loaded")

    @property
    def device(self):
        return next(self.parameters()).device

    def count_parameters(self):
        """Count the number of trainable parameters in the model."""
        from prettytable import PrettyTable
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, module in self.named_children():
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)

            if getattr(self, "tie_embedding", False) and name == "softmax":
                table.add_row([name, "tied"])
            else:
                table.add_row([name, f"{params:,d}"])
                total_params += params
        # add separation line
        table.add_row(["-" * 10, "-" * 10])
        table.add_row(["Total", f"{total_params:,d}"])
        print(table)
        return total_params
