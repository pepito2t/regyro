"""Tracks which datasets a shard directory holds and under what licence.

UZH-FPV ships under CC BY-NC-SA, so a model trained on it inherits non-commercial
and share-alike terms. Whether a checkpoint can be published is a property of the
data it saw, and that is only knowable if it is recorded at import time.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
OWN_FOOTAGE_LICENSE = "own-footage"
UNKNOWN_LICENSE = "unknown"

# Substrings that mark a licence as restricting commercial use or requiring share-alike.
NON_COMMERCIAL_MARKERS = ("nc", "non-commercial", "noncommercial", "research-only")
SHARE_ALIKE_MARKERS = ("sa", "share-alike", "sharealike")


@dataclass(frozen=True)
class Provenance:
    sources: list[str]
    licenses: list[str]

    @property
    def restricted_licenses(self) -> list[str]:
        return [name for name in self.licenses if _has_marker(name, NON_COMMERCIAL_MARKERS)]

    @property
    def share_alike_licenses(self) -> list[str]:
        return [name for name in self.licenses if _has_marker(name, SHARE_ALIKE_MARKERS)]

    @property
    def is_publishable(self) -> bool:
        """True when nothing in the training data blocks unrestricted redistribution."""
        return not self.restricted_licenses and UNKNOWN_LICENSE not in self.licenses


def _has_marker(license_name: str, markers: tuple[str, ...]) -> bool:
    tokens = {token for token in license_name.lower().replace("_", "-").split("-") if token}
    return any(marker in tokens or marker in license_name.lower() for marker in markers)


def read_manifest(shard_dir: Path) -> list[dict]:
    path = shard_dir / MANIFEST_NAME
    if not path.exists():
        return []
    try:
        entries = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("cannot read %s: %s", path, exc)
        return []
    return entries if isinstance(entries, list) else []


def summarise(entries: list[dict]) -> Provenance:
    sources = sorted({str(entry.get("source", "?")) for entry in entries})
    licenses = sorted({str(entry.get("license", UNKNOWN_LICENSE)) for entry in entries})
    return Provenance(sources=sources, licenses=licenses)


def describe(shard_dir: Path) -> Provenance:
    return summarise(read_manifest(shard_dir))
