"""The word edit distance that every error count of the bench scripts uses."""


def edits(ref: list[str], hyp: list[str]) -> int:
    """Insertions, deletions, and substitutions that turn `ref` into `hyp`."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref):
        cur = [i + 1] + [0] * len(hyp)
        for j, h in enumerate(hyp):
            cur[j + 1] = min(prev[j] + (r != h), prev[j + 1] + 1, cur[j] + 1)
        prev = cur
    return prev[-1]
