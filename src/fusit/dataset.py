"""
Dataset loaders for the corpora vendored under `dataset/`.

Each corpus is a `Dataset` subclass exposing the same contract, so callers that handle
more than one can stay dataset-agnostic:

    >>> from fusit.dataset import get_dataset, DATASETS
    >>> ds = get_dataset("synthpai")
    >>> items = ds.select(n=50, seed=0)
    >>> items[0].username, list(items[0].relevant_pii)
    ('BlueberryBiscuit', ['age', 'occupation', 'city_country'])

Every item carries `.username`, `.text` and `.relevant_pii` (attribute -> ground-truth
value); the datasets differ only in what an item *is* and in how ground truth is filtered
(see each class's docstring).

Data files live in `dataset/` at the repo root, resolved relative to this file, so an
editable install needs no configuration. A built wheel does not ship them (they sit
outside `src/`), so set FUSIT_DATASET_DIR to point at them there; a missing file raises
with that instruction rather than a bare FileNotFoundError.

Run `python -m fusit.dataset` for corpus statistics.
"""

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Type

__all__ = [
    "DATASET_DIR",
    "DATASETS",
    "Dataset",
    "Item",
    "SynthPAI",
    "Synthetic",
    "SyntheticItem",
    "get_dataset",
    "load_dataset",
]

DATASET_DIR = Path(
    os.environ.get("FUSIT_DATASET_DIR", Path(__file__).resolve().parents[2] / "dataset")
)


@dataclass
class Item:
    """One evaluation unit: text plus the ground-truth attributes it leaks."""

    username: str  # stable per-item id
    text: str
    relevant_pii: Dict[str, str] = field(default_factory=dict)  # attribute -> ground truth


@dataclass
class SyntheticItem(Item):
    hardness: int = 0


class Dataset:
    """Base contract. Subclasses implement `load()`; everything else is shared.

    Instances cache the parsed corpus, so `get_dataset(name)` (which memoizes) parses each
    jsonl at most once per process even when callers ask for several subsamples.
    """

    name: ClassVar[str]
    filename: ClassVar[str]

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else DATASET_DIR / self.filename
        self._cache: Optional[List[Item]] = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(path={str(self.path)!r})"

    # -- reading ---------------------------------------------------------------

    def _records(self) -> List[dict]:
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.name} data not found at {self.path}. Set FUSIT_DATASET_DIR to the "
                f"directory holding {self.filename}, or see dataset/README.md."
            )
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def load(self) -> List[Item]:
        """Parse every usable item. Subclass responsibility."""
        raise NotImplementedError

    def items(self) -> List[Item]:
        """`load()`, memoized on the instance."""
        if self._cache is None:
            self._cache = self.load()
        return self._cache

    # -- sampling --------------------------------------------------------------

    def select(self, n: Optional[int] = None, seed: int = 0) -> List[Item]:
        """Fixed-seed subsample, or the whole corpus when `n` is None or >= its size.
        Always a fresh list, so callers may reorder it without disturbing the cache."""
        items = self.items()
        if n is not None and n < len(items):
            return random.Random(seed).sample(items, n)
        return list(items)

    # -- reporting -------------------------------------------------------------

    def stats(self) -> Dict:
        from collections import Counter

        items = self.items()
        total = sum(len(i.text) for i in items)
        return {
            "num_items": len(items),
            "total_chars": total,
            "avg_chars": total / len(items) if items else 0,
            "attribute_counts": dict(Counter(a for i in items for a in i.relevant_pii)),
        }


class SynthPAI(Dataset):
    """SynthPAI profiles (Phase 4: implicit-PII / cue-tagger reproduction).

    One item = one *profile*, text = all its comments joined with "\\n" (same as TRACE-RPS's
    trace.py::run_reddit_anonymization). Reads the vendored ground-truth jsonl directly
    rather than importing TRACE-RPS (its anonymization/trace.py assumes cwd=anonymization/
    and does bare `from prompts import ...`, brittle to import cross-repo).

    Record shape: {"username", "comments": [...], "reviews": {"human": {attr: {"estimate",
    "hardness", "certainty"}, ...}, "human_evaluated": {...}}}. We use "human" (the raw
    annotation), not "human_evaluated" (TRACE's own model-vs-gt scoring artifact, not
    ground truth itself).

    "Relevant" PII mirrors TRACE-RPS's Profile.get_relevant_pii
    (src/reddit/reddit_types.py): hardness >= 1 and certainty >= 1. Checked empirically:
    all 298/298 profiles qualify on >=1 attribute under this filter (2026-09-02), so no
    profile is dropped by this criterion alone — it only trims which per-profile attributes
    count as eval targets.
    """

    name = "synthpai"
    filename = "synthpai.jsonl"

    # TRACE-RPS's SENSITIVE_ATTRIBUTES (anonymization/trace.py) mapped to the raw
    # ground-truth keys actually present in reviews.human (via anonymization/utils.py's
    # map_synthpai_to_pii, inverted). "time"/"timestamp" are bookkeeping, not attributes.
    ATTRIBUTES: ClassVar[List[str]] = [
        "age",
        "sex",
        "relationship_status",
        "education",
        "occupation",
        "income_level",
        "city_country",
        "birth_city_country",
    ]

    @classmethod
    def _relevant_pii(cls, review_human: dict) -> Dict[str, str]:
        out = {}
        for attr in cls.ATTRIBUTES:
            entry = review_human.get(attr)
            if not isinstance(entry, dict):
                continue
            if (
                entry.get("hardness", 0) >= 1
                and entry.get("certainty", 0) >= 1
                and entry.get("estimate")
            ):
                out[attr] = entry["estimate"]
        return out

    def load(self) -> List[Item]:
        items = []
        for r in self._records():
            rel = self._relevant_pii(r["reviews"].get("human", {}))
            if not rel:
                continue
            text = "\n".join(c["text"] for c in r["comments"])
            items.append(Item(username=r["username"], text=text, relevant_pii=rel))
        return items

    def stats(self) -> Dict:
        items = self.items()
        return {
            **super().stats(),
            "avg_relevant_attrs": (
                sum(len(i.relevant_pii) for i in items) / len(items) if items else 0
            ),
        }


class Synthetic(Dataset):
    """TRACE-RPS's synthetic comment corpus — the second dataset of Experiment Plan 2.1.

    Unit differs from SynthPAI: one item = one *comment* with exactly ONE target attribute
    (`feature`), not a profile with many. Ground truth is `personality[feature]`.

    Protocol mirrors TRACE-RPS's `src/reddit/reddit_utils.py::load_synthetic_profile` (read
    verbatim, not imported — that module does bare `from src...` and pulls in the whole
    Profile/Comment stack):

    - text = record["response"]. TRACE splits it on "\\n" into Comment objects and later
      re-joins them with "\\n"; the round trip only drops blank lines, so we join the
      non-blank lines directly and skip the dataclass detour.
    - certainty is 5 by construction (TRACE hardcodes it) and hardness = record["hardness"],
      so SynthPAI's `hardness>=1 & certainty>=1` filter admits every record here and is not
      re-applied.
    - records with feature == "income" are dropped ("We only deal with income_level" —
      TRACE's comment). No such record exists in this file (checked: 0/525); kept for parity.

    Attribute KEYS stay in SynthPAI's vocabulary (`income_level`/`sex`/`city_country`)
    rather than TRACE's display names (`income`/`gender`/`location`), so repro/cue_tagger.py
    and repro/synthpai_eval.py work across both datasets unchanged; cue_tagger's
    ATTRIBUTE_LABEL already maps key -> display name when building prompts.
    """

    name = "synthetic"
    filename = "synthetic_dataset.jsonl"

    def load(self) -> List[Item]:
        items = []
        for idx, r in enumerate(self._records()):
            feature = r["feature"]
            if feature == "income":  # TRACE: "We only deal with income_level"
                continue
            gt = r["personality"].get(feature)
            if not gt:
                continue
            text = "\n".join(s for s in r["response"].split("\n") if s.strip())
            if not text.strip():
                continue
            items.append(
                SyntheticItem(
                    # TRACE's "<age><sex>" id is NOT unique; index by record instead
                    username=f"syn{idx:04d}-{feature}",
                    text=text,
                    relevant_pii={feature: str(gt)},
                    hardness=int(r.get("hardness", 0)),
                )
            )
        return items

    def stats(self) -> Dict:
        from collections import Counter

        return {
            **super().stats(),
            "hardness_counts": dict(Counter(i.hardness for i in self.items())),
        }


_REGISTRY: Dict[str, Type[Dataset]] = {cls.name: cls for cls in (SynthPAI, Synthetic)}
DATASETS = tuple(_REGISTRY)

_instances: Dict[str, Dataset] = {}


def get_dataset(name: str) -> Dataset:
    """The `Dataset` for `name`, memoized so its jsonl is parsed once per process."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown dataset {name!r}, expected one of {DATASETS}")
    if name not in _instances:
        _instances[name] = _REGISTRY[name]()
    return _instances[name]


def load_dataset(name: str, n: Optional[int] = None, seed: int = 0) -> List[Item]:
    """One-shot `get_dataset(name).select(n, seed)`."""
    return get_dataset(name).select(n=n, seed=seed)


if __name__ == "__main__":
    import sys

    for name in sys.argv[1:] or DATASETS:
        ds = get_dataset(name)
        print(f"=== {name} ({ds.path}) ===")
        for k, v in ds.stats().items():
            print(f"  {k}: {v}")
