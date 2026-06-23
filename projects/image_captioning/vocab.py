import re
from collections import Counter
from dataclasses import dataclass, field

TOKEN_PATTERN = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")


def tokenize(text: str) -> list[str]:
    """Return lowercase word tokens from a caption."""
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class Vocabulary:
    """Small caption vocabulary with special tokens and frequency filtering."""

    min_freq: int = 1
    pad_token: str = '<pad>'
    start_token: str = '<start>'
    end_token: str = '<end>'
    unk_token: str = '<unk>'
    token_to_idx: dict[str, int] = field(default_factory=dict)
    idx_to_token: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.min_freq < 1:
            raise ValueError('min_freq must be at least 1')
        if not self.token_to_idx:
            for token in [self.pad_token, self.start_token, self.end_token, self.unk_token]:
                self._add_token(token)

    def __len__(self) -> int:
        return len(self.token_to_idx)

    @property
    def pad_idx(self) -> int:
        return self.token_to_idx[self.pad_token]

    @property
    def start_idx(self) -> int:
        return self.token_to_idx[self.start_token]

    @property
    def end_idx(self) -> int:
        return self.token_to_idx[self.end_token]

    @property
    def unk_idx(self) -> int:
        return self.token_to_idx[self.unk_token]

    def _add_token(self, token: str) -> None:
        if token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token

    def fit(self, captions: list[str]) -> None:
        counter: Counter[str] = Counter()
        for caption in captions:
            counter.update(tokenize(caption))

        for token, count in sorted(counter.items()):
            if count >= self.min_freq:
                self._add_token(token)

    def encode(self, caption: str, add_special_tokens: bool = True) -> list[int]:
        indices = [self.token_to_idx.get(token, self.unk_idx) for token in tokenize(caption)]
        if add_special_tokens:
            return [self.start_idx] + indices + [self.end_idx]
        return indices

    def decode(self, indices: list[int], skip_special_tokens: bool = True) -> str:
        special = {self.pad_token, self.start_token, self.end_token, self.unk_token}
        tokens = [self.idx_to_token.get(int(idx), self.unk_token) for idx in indices]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in special]
        return ' '.join(tokens)
