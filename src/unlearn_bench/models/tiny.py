from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import nn


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass(frozen=True)
class Vocabulary:
    tokens: tuple[str, ...]

    @property
    def stoi(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def encode_prompt(self, text: str) -> list[int]:
        mapping = self.stoi
        return [mapping.get(token, mapping["<unk>"]) for token in _tokens(text)]

    def encode_completion(self, text: str) -> int:
        return self.stoi.get(text.lower(), self.stoi["<unk>"])


def build_vocabulary(records: list[dict]) -> Vocabulary:
    values = {"<unk>"}
    for row in records:
        values.update(_tokens(row["prompt"]))
        values.add(row["completion"].lower())
        if row.get("replacement"):
            values.add(row["replacement"].lower())
    return Vocabulary(tuple(["<unk>"] + sorted(values - {"<unk>"})))


class TinyAssociationLM(nn.Module):
    """Small bag-of-tokens next-token model used only for CI and pipeline calibration."""

    def __init__(self, vocab_size: int, hidden_size: int = 24) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, prompts: list[list[int]]) -> torch.Tensor:
        device = self.embedding.weight.device
        vectors = []
        for prompt in prompts:
            ids = torch.tensor(prompt, device=device, dtype=torch.long)
            vectors.append(self.embedding(ids).mean(dim=0))
        return self.output(torch.stack(vectors))
