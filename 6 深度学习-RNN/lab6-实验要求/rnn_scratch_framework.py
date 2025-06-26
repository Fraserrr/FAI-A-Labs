"""
RNN From‑Scratch Framework（选做）
==========================
A pure‑NumPy implementation skeleton for a character‑level recurrent neural
network.  No deep‑learning libraries are used – students must complete missing
parts (marked TODO) to build forward and backward passes, train the model, and
sample text.

Run with:
    python rnn_scratch_framewor.py
"""

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests

# -----------------------------------------------------------------------------
# 1. Command‑line arguments
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="NumPy RNN Shakespeare")
    p.add_argument("--seq_len", type=int, default=40, help="sequence length")
    p.add_argument("--hidden", type=int, default=128, help="hidden size")
    p.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    p.add_argument("--epochs", type=int, default=10, help="training epochs")
    p.add_argument("--batch", type=int, default=64, help="mini‑batch size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data", default="shakespeare.txt")
    return p.parse_args()


# -----------------------------------------------------------------------------
# 2. Utils
# -----------------------------------------------------------------------------


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def download_corpus(path):
    if Path(path).exists():
        return
    url = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
        "tinyshakespeare/input.txt"
    )
    print("[+] downloading corpus …")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    Path(path).write_text(r.text, encoding="utf-8")
    print("[✓] saved", path)


# -----------------------------------------------------------------------------
# 3. Data preparation
# -----------------------------------------------------------------------------

def build_dataset(text, seq_len):
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for c, i in c2i.items()}
    vocab = len(chars)

    # encode entire corpus as int array
    encoded = np.array([c2i[c] for c in text], dtype=np.int32)

    # create sequences + targets
    X, y = [], []
    for i in range(0, len(encoded) - seq_len):
        X.append(encoded[i : i + seq_len])
        y.append(encoded[i + seq_len])
    X = np.stack(X)  # (N, seq_len)
    y = np.array(y, dtype=np.int32)  # (N,)
    return X, y, vocab, c2i, i2c


# -----------------------------------------------------------------------------
# 4. Model helpers
# -----------------------------------------------------------------------------

def one_hot(indices, depth):
    """Return 2‑D one‑hot array."""
    out = np.zeros((indices.size, depth), dtype=np.float32)
    out[np.arange(indices.size), indices.flatten()] = 1.0
    return out.reshape(*indices.shape, depth)


class RNNCell:
    """Single‑timestep vanilla RNN cell (tanh)."""

    def __init__(self, input_size, hidden_size):
        scale = 1.0 / np.sqrt(hidden_size)
        self.Wxh = np.random.randn(input_size, hidden_size) * scale
        self.Whh = np.random.randn(hidden_size, hidden_size) * scale
        self.bh = np.zeros(hidden_size)

        # gradients (buffered during BPTT)
        self.dWxh = np.zeros_like(self.Wxh)
        self.dWhh = np.zeros_like(self.Whh)
        self.dbh = np.zeros_like(self.bh)

    def forward(self, x_t, h_prev):
        """x_t: (B, I), h_prev: (B, H) → h_t, cache"""
        h_raw = x_t @ self.Wxh + h_prev @ self.Whh + self.bh
        h_t = np.tanh(h_raw)
        cache = (x_t, h_prev, h_t)
        return h_t, cache

    def backward(self, dh_t, cache):
        """Compute parameter gradients for one step.
        dh_t: upstream gradient w.r.t. h_t (B, H)
        Returns dh_prev to propagate backwards in time.
        """
        x_t, h_prev, h_t = cache
        dtanh = dh_t * (1.0 - h_t ** 2)  # (B, H)

        # parameter grads accumulate over sequence
        self.dWxh += x_t.T @ dtanh
        self.dWhh += h_prev.T @ dtanh
        self.dbh += dtanh.sum(axis=0)

        dh_prev = dtanh @ self.Whh.T
        return dh_prev

    def zero_grad(self):
        self.dWxh.fill(0)
        self.dWhh.fill(0)
        self.dbh.fill(0)

    def step_grad(self, lr):
        for param, grad in (
            (self.Wxh, self.dWxh),
            (self.Whh, self.dWhh),
            (self.bh, self.dbh),
        ):
            param -= lr * grad


class RNN:
    """Character‑level RNN with softmax output (single hidden layer)."""

    def __init__(self, vocab, hidden):
        self.cell = RNNCell(vocab, hidden)
        # output weights
        scale = 1.0 / np.sqrt(hidden)
        self.Why = np.random.randn(hidden, vocab) * scale
        self.by = np.zeros(vocab)
        # grads
        self.dWhy = np.zeros_like(self.Why)
        self.dby = np.zeros_like(self.by)

    def zero_grad(self):
        self.cell.zero_grad()
        self.dWhy.fill(0)
        self.dby.fill(0)

    # ---------------------------------------------------------------------
    # TODO 1: implement forward pass over an entire sequence.
    # Inputs:
    #   X_batch: (B, T) int indices
    #   h0:      (B, H) initial hidden state
    # Should return (logits, h_T, caches)
    #   logits:  (B, V) scores for final time‑step
    #   caches:  list of step caches for backprop
    # ---------------------------------------------------------------------
    def forward(self, X_batch, h0):
        B, T = X_batch.shape
        V = self.by.size
        H = h0.shape[1]
        h_t = h0
        caches = []
        for t in range(T):
            x_t = one_hot(X_batch[:, t], V)  # (B, V)
            h_t, cache = self.cell.forward(x_t, h_t)
            caches.append(cache)
        logits = h_t @ self.Why + self.by  # (B, V)
        return logits, h_t, caches

    # ---------------------------------------------------------------------
    # TODO 2: backward through time.
    # Inputs:
    #   dlogits: (B, V) gradient on scores
    #   caches:  list returned by forward
    #   h_T:     final hidden state
    # Returns dh0 (gradient wrt initial hidden)
    # ---------------------------------------------------------------------
    def backward(self, dlogits, caches, h_T):
        B, V = dlogits.shape
        H = h_T.shape[1]
        # output layer grads
        self.dWhy += h_T.T @ dlogits
        self.dby += dlogits.sum(axis=0)
        dh = dlogits @ self.Why.T  # (B, H)

        # back‑prop through time
        for cache in reversed(caches):
            dh = self.cell.backward(dh, cache)
        dh0 = dh
        return dh0

    def step_grad(self, lr):
        self.cell.step_grad(lr)
        for param, grad in ((self.Why, self.dWhy), (self.by, self.dby)):
            param -= lr * grad


# -----------------------------------------------------------------------------
# 5. Loss helpers
# -----------------------------------------------------------------------------

def softmax_cross_entropy(logits, targets):
    """Compute loss and gradient (vectorised). targets = int indices (B,)"""
    logits = logits - logits.max(axis=1, keepdims=True)  # numerical stability
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    N = targets.size
    loss = -np.log(probs[np.arange(N), targets]).mean()
    grad = probs
    grad[np.arange(N), targets] -= 1.0
    grad /= N
    return loss, grad


# -----------------------------------------------------------------------------
# 6. Training loop
# -----------------------------------------------------------------------------

def train(model, X, y, vocab, epochs, lr, batch_size, hidden_size):
    N = X.shape[0]
    history = []
    for ep in range(1, epochs + 1):
        idx = np.random.permutation(N)
        X, y = X[idx], y[idx]
        ep_loss = 0.0
        for i in range(0, N, batch_size):
            xb = X[i : i + batch_size]
            yb = y[i : i + batch_size]
            if xb.shape[0] != batch_size:
                continue  # drop last small batch for simplicity

            h0 = np.zeros((batch_size, hidden_size), dtype=np.float32)
            logits, h_T, caches = model.forward(xb, h0)
            loss, dlogits = softmax_cross_entropy(logits, yb)
            model.zero_grad()
            model.backward(dlogits, caches, h_T)
            model.step_grad(lr)
            ep_loss += loss * xb.shape[0]
        ep_loss /= N
        history.append(ep_loss)
        print(f"epoch {ep}/{epochs}  loss {ep_loss:.4f}")
    return history


# -----------------------------------------------------------------------------
# 7. Sampling
# -----------------------------------------------------------------------------

def sample(model, seed, c2i, i2c, vocab, length=200, temp=1.0, hidden_size=128):
    seq = [c2i[c] for c in seed]
    h = np.zeros((1, hidden_size), dtype=np.float32)
    out = seed
    for _ in range(length):
        x = np.array(seq[-args.seq_len :], dtype=np.int32).reshape(1, -1)
        logits, h, _ = model.forward(x, h)
        logits /= temp
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        idx = np.random.choice(vocab, p=probs.ravel())
        out += i2c[idx]
        seq.append(idx)
    return out


# -----------------------------------------------------------------------------
# 8. Main
# -----------------------------------------------------------------------------


def main():
    global args
    args = parse_args()
    set_seed(args.seed)
    download_corpus(args.data)

    text = Path(args.data).read_text(encoding="utf-8")
    X, y, vocab, c2i, i2c = build_dataset(text, args.seq_len)

    model = RNN(vocab, args.hidden)
    losses = train(
        model, X, y, vocab, args.epochs, args.lr, args.batch, args.hidden
    )

    plt.figure()
    plt.plot(losses)
    plt.title("training loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    fig = f"loss_numpy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(fig)
    print("[✓] curve saved →", fig)

    print("\n--- sample ---")
    print(sample(model, "ROMEO: ", c2i, i2c, vocab, hidden_size=args.hidden))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
