"""Config loading with dict-to-attribute access."""
from __future__ import annotations
import yaml


class Config(dict):
    """Dict that also supports attribute access, recursively."""

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[key] = value
        return value

    def __setattr__(self, key, value):
        self[key] = value


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(raw)
