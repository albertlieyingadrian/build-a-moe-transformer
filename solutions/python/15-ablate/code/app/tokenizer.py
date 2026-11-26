"""Characters to numbers and back (stage 01)."""

from __future__ import annotations

import torch


class CharTokenizer:
    """The vocabulary is every distinct character in the corpus, sorted; a character's id is its place in that list."""

    def __init__(self, corpus: str):
        self.chars = sorted(set(corpus))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.chars[i] for i in ids)


def split(ids: list[int], val_fraction: float = 0.1):
    """The first 90% of the text trains the model; the last 10% checks it on text it never saw."""
    data = torch.tensor(ids, dtype=torch.long)
    n = int(len(data) * (1 - val_fraction))
    return data[:n], data[n:]


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, gen: torch.Generator):
    """batch_size random windows; the target is the same window shifted one character to the right."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y
