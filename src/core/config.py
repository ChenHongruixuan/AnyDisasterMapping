"""YAML-based configuration for the unified deep-learning trainer."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Thin wrapper around a flat YAML dict with dot-notation access."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = data or {}
        self.resume: str | None = None

    # -- construction --------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a YAML file and return a Config instance."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls(raw)

    # -- attribute access ----------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name == "resume":
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"Config has no key '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name == "resume":
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    # -- safe access ---------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by key. Supports dotted paths like 'loss.use_lovasz'."""
        parts = key.split(".")
        cur: Any = self._data
        for part in parts:
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    # -- overrides -----------------------------------------------------------

    def apply_overrides(self, args: list[str]) -> None:
        """Apply CLI-style overrides: ["key", "value", "key2", "value2", ...]."""
        it = iter(args)
        for key in it:
            raw_value = next(it)
            value = self._cast(raw_value)
            parts = key.split(".")
            target = self._data
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value

    # -- serialisation -------------------------------------------------------

    def save_yaml(self, path: str | Path) -> None:
        """Dump the current config to a YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self._data, f, default_flow_style=False, sort_keys=False)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _cast(value: str) -> Any:
        """Auto-cast a string to int, float, bool, None, or leave as str."""
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        if value.lower() in ("null", "none"):
            return None
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy of the underlying data dict."""
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"Config({self._data!r})"
