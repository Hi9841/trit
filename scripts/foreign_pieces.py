# /// script
# requires-python = ">=3.10"
# dependencies = ["tokenizers", "wordfreq"]
# ///
"""Writes src/text/foreign-pieces.txt, which Trit embeds.

usage: uv run trit/scripts/foreign_pieces.py <tokenizer.json>

The file holds the pieces of the Parakeet v3 tokenizer that the 30000 most
frequent words of 23 other languages use and the 50000 most frequent English
words (wordfreq) do not.
"""

import sys
from pathlib import Path

from tokenizers import Tokenizer
from wordfreq import top_n_list

OUT = Path(__file__).resolve().parents[1] / "src" / "text" / "foreign-pieces.txt"
OTHERS = ["fr", "es", "de", "it", "pt", "nl", "pl", "sv", "da", "fi", "cs", "ro",
          "hu", "sk", "sl", "sh", "bg", "el", "lv", "lt", "ru", "uk", "nb"]


def pieces(tokenizer: Tokenizer, vocabulary: list[str]) -> set[str]:
    out = set()
    for word in vocabulary:
        for variant in {word, word.lower(), word.capitalize(), word.upper()}:
            out.update(tokenizer.encode(variant, add_special_tokens=False).tokens)
    return out


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    tokenizer = Tokenizer.from_file(sys.argv[1])
    english = pieces(tokenizer, top_n_list("en", 50000))
    others = set()
    for language in OTHERS:
        others |= pieces(tokenizer, top_n_list(language, 30000))
    ascii_letters = lambda p: p.isascii() or all(c.isascii() or c == "▁" for c in p)
    foreign = sorted(p for p in tokenizer.get_vocab() if p in others and p not in english
                     and any(c.isalpha() for c in p) and ascii_letters(p))
    OUT.write_text("\n".join(foreign) + "\n", encoding="utf-8", newline="\n")
    print(f"foreign-pieces.txt: {len(foreign)} pieces")


if __name__ == "__main__":
    main()
