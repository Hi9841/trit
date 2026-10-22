# /// script
# requires-python = ">=3.10"
# dependencies = ["tokenizers", "wordfreq"]
# ///
"""Writes the two English-only lists that Trit embeds (src/text/).

usage: uv run trit/scripts/english_lists.py <american-english-large> <tokenizer.json>

- english-words.txt: the lowercase words of SCOWL size 70 (Debian package
  wamerican-large), apostrophes removed, front-coded: each line starts with
  one character for the letters shared with the word before (0-9, then a-z
  for 10-35), then the rest of the word.
- foreign-pieces.txt: pieces of the Parakeet v3 tokenizer that the 30000
  most frequent words of 23 other languages use and the 50000 most
  frequent English words (wordfreq) do not.
"""

import re
import sys
from pathlib import Path

from tokenizers import Tokenizer
from wordfreq import top_n_list

OUT = Path(__file__).resolve().parents[1] / "src" / "text"
DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"
OTHERS = ["fr", "es", "de", "it", "pt", "nl", "pl", "sv", "da", "fi", "cs", "ro",
          "hu", "sk", "sl", "sh", "bg", "el", "lv", "lt", "ru", "uk", "nb"]


def words(scowl: str) -> list[str]:
    out = set()
    for line in Path(scowl).read_text(encoding="latin-1").splitlines():
        word = line.strip().replace("'", "").lower()
        if re.fullmatch(r"[a-z]+", word):
            out.add(word)
    return sorted(out)


def front_coded(sorted_words: list[str]) -> str:
    lines, previous = [], ""
    for word in sorted_words:
        shared = 0
        while shared < min(len(word), len(previous), 35) and word[shared] == previous[shared]:
            shared += 1
        lines.append(DIGITS[shared] + word[shared:])
        previous = word
    return "\n".join(lines) + "\n"


def pieces(tokenizer: Tokenizer, vocabulary: list[str]) -> set[str]:
    out = set()
    for word in vocabulary:
        for variant in {word, word.lower(), word.capitalize(), word.upper()}:
            out.update(tokenizer.encode(variant, add_special_tokens=False).tokens)
    return out


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    (OUT / "english-words.txt").write_text(front_coded(words(sys.argv[1])), encoding="utf-8", newline="\n")
    tokenizer = Tokenizer.from_file(sys.argv[2])
    english = pieces(tokenizer, top_n_list("en", 50000))
    others = set()
    for language in OTHERS:
        others |= pieces(tokenizer, top_n_list(language, 30000))
    ascii_letters = lambda p: p.isascii() or all(c.isascii() or c == "▁" for c in p)
    foreign = sorted(p for p in tokenizer.get_vocab() if p in others and p not in english
                     and any(c.isalpha() for c in p) and ascii_letters(p))
    (OUT / "foreign-pieces.txt").write_text("\n".join(foreign) + "\n", encoding="utf-8", newline="\n")
    print(f"english-words.txt, foreign-pieces.txt: {len(foreign)} pieces")


if __name__ == "__main__":
    main()
