"""Tiny synthetic character-level corpus for toy draft/target LMs."""
from __future__ import annotations

from typing import List, Tuple

# Short, repetitive English-ish phrases so n-grams have signal and a weak
# bigram draft still proposes something the stronger trigram often accepts.
PHRASES = [
    "the cat sat on the mat",
    "the dog sat on the rug",
    "a cat sat on a mat",
    "a dog ran on the mat",
    "the cat ran on the rug",
    "the dog sat on the mat",
    "a cat ran on the mat",
    "the fox sat on the log",
    "a fox ran on the log",
    "the fox sat on the mat",
    "she saw the cat on the mat",
    "he saw the dog on the rug",
    "she saw a fox on the log",
    "he saw the cat on the rug",
    "the boy saw the dog",
    "the girl saw the cat",
    "the boy saw a fox",
    "the girl saw the dog",
    "one cat sat on one mat",
    "two dogs sat on the rug",
    "one fox ran on the log",
    "two cats ran on the mat",
    "the red cat sat on the mat",
    "the big dog sat on the rug",
    "a red fox ran on the log",
    "a big cat ran on the mat",
    "see the cat on the mat",
    "see the dog on the rug",
    "see a fox on the log",
    "see the red cat on the mat",
]


def build_corpus(repeats: int = 40) -> Tuple[str, List[str]]:
    """Concatenate phrases with spaces; return (full_text, vocab_chars)."""
    body = " ".join(PHRASES * repeats)
    # space + lowercase letters that appear
    chars = sorted(set(body))
    return body, chars


def encode(text: str, stoi: dict) -> List[int]:
    return [stoi[c] for c in text]


def decode(ids: List[int], itos: dict) -> str:
    return "".join(itos[i] for i in ids)


def make_maps(chars: List[str]) -> Tuple[dict, dict]:
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos
