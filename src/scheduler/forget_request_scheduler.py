"""
ForgetRequestScheduler — splits a single forget10 pool into T sequential
forget requests, so unlearning runs as genuine continual unlearning
(request 1, then request 2, ...) instead of repeating rounds on the same
fixed 200-example set.

Groups by author when meta['author'] (or author_id/author_name) is
present -- "forget everything about these N authors" is a semantically
coherent unit. Falls back to a naive shuffled chunk split if no author
metadata is found.
"""
from __future__ import annotations
from dataclasses import dataclass, field


def _get_meta(ex):
    return ex.get("meta") if isinstance(ex, dict) else getattr(ex, "meta", None)


def _author_key(ex):
    meta = _get_meta(ex)
    if not meta:
        return None
    for key in ("author", "author_id", "author_name"):
        if key in meta:
            return meta[key]
    return None


@dataclass
class ForgetRequest:
    request_id: int
    examples: list
    author_ids: list = field(default_factory=list)

    def __len__(self):
        return len(self.examples)


class ForgetRequestScheduler:
    def __init__(self, forget_examples: list, num_requests: int = 5, seed: int = 42):
        self.forget_examples = forget_examples
        self.num_requests = num_requests
        self.seed = seed
        self.requests: list[ForgetRequest] = self._build_requests()

    def _build_requests(self) -> list[ForgetRequest]:
        import random

        author_groups: dict = {}
        ungrouped = []
        for ex in self.forget_examples:
            key = _author_key(ex)
            if key is None:
                ungrouped.append(ex)
            else:
                author_groups.setdefault(key, []).append(ex)

        rng = random.Random(self.seed)

        if author_groups and not ungrouped:
            authors = list(author_groups.keys())
            rng.shuffle(authors)

            buckets = [[] for _ in range(self.num_requests)]
            bucket_sizes = [0] * self.num_requests
            author_buckets = [[] for _ in range(self.num_requests)]

            for author in authors:
                target = min(range(self.num_requests), key=lambda i: bucket_sizes[i])
                buckets[target].extend(author_groups[author])
                author_buckets[target].append(author)
                bucket_sizes[target] += len(author_groups[author])

            return [
                ForgetRequest(request_id=i, examples=buckets[i], author_ids=author_buckets[i])
                for i in range(self.num_requests) if buckets[i]
            ]

        # Fallback: no (or mixed) author metadata -- naive chunk split.
        examples = list(self.forget_examples)
        rng.shuffle(examples)
        chunk_size = max(1, len(examples) // self.num_requests)
        chunks = [examples[i:i + chunk_size] for i in range(0, len(examples), chunk_size)]
        if len(chunks) > self.num_requests:
            chunks[-2].extend(chunks[-1])
            chunks = chunks[:-1]

        return [ForgetRequest(request_id=i, examples=chunks[i]) for i in range(len(chunks))]

    def __iter__(self):
        return iter(self.requests)

    def __len__(self):
        return len(self.requests)
