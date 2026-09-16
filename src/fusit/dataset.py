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
from typing import ClassVar, Dict, List, Optional, Sequence, Type

__all__ = [
    "ATTACK_ENTITY_TYPES",
    "DATASET_DIR",
    "DATASETS",
    "Dataset",
    "Item",
    "PRIVATE_IDENTIFIER_TYPES",
    "Span",
    "TAB_ENTITY_TYPES",
    "TabDocument",
    "TabECHR",
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


@dataclass
class Span:
    """One annotated private span: `text[start:end]`, typed by TAB's entity taxonomy."""

    start: int
    end: int
    entity_type: str
    text: str


@dataclass
class TabDocument:
    """A TAB-ECHR document.

    Deliberately NOT an `Item`. The two corpora answer different questions: an `Item` says
    "this text leaks *these attribute values*" and carries `relevant_pii`; a `TabDocument`
    says "*these characters* are private" and carries typed spans. There is no attribute ->
    value map here to fill in, and pretending otherwise would put an empty dict on every
    document and quietly break anything that trusts `relevant_pii`.

    What they do share is `username`/`text`, so code that only needs an id and a body works
    on both.
    """

    username: str  # TAB's doc_id
    text: str
    spans: List[Span] = field(default_factory=list)

    @property
    def doc_id(self) -> str:
        """TAB's own name for the id, for readers coming from the benchmark."""
        return self.username

    def spans_of(self, entity_types: Optional[Sequence[str]] = None) -> List[Span]:
        """Spans, optionally restricted to some entity types."""
        if entity_types is None:
            return list(self.spans)
        allowed = set(entity_types)
        return [s for s in self.spans if s.entity_type in allowed]

    def offsets(self, entity_types: Optional[Sequence[str]] = None) -> List[List[int]]:
        """`[[start, end], ...]` -- the span form the rest of the package speaks."""
        return [[s.start, s.end] for s in self.spans_of(entity_types)]

    def entity_types(self) -> List[str]:
        return sorted({s.entity_type for s in self.spans})


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


#: The eight categories TAB annotates and DP-Fusion treats as private (paper Table 3).
#: Same list as `fusit.utils.ENTITY_TYPES`, which is also how `DEFAULT_BETA_DICT` is keyed.
TAB_ENTITY_TYPES = ["PERSON", "CODE", "LOC", "ORG", "DEM", "DATETIME", "QUANTITY", "MISC"]

#: Paper Section 5.1: every type is private during generation, but the *attack* is scored
#: only on these three, "as they appear consistently across all documents".
ATTACK_ENTITY_TYPES = ["PERSON", "CODE", "DATETIME"]

#: Paper Appendix A.5: "we do not distinguish between direct and quasi identifiers. Instead,
#: we take their union and treat all such values uniformly". NO_MASK is the third value TAB
#: uses and marks spans the annotators judged safe to leave in.
PRIVATE_IDENTIFIER_TYPES = {"DIRECT", "QUASI"}


class TabECHR(Dataset):
    """TAB-ECHR: European Court of Human Rights judgments with hand-annotated private spans.

    This is DP-Fusion's own evaluation corpus (paper Section 5.1), and the reason its item
    type differs from the other two: the ground truth is *which characters are private*, not
    which attribute value can be guessed. See `TabDocument`.

    Two annotation choices have to be made when reading the file, and both are recorded here
    rather than buried in a caller:

    - **Which identifier types count.** TAB marks each mention DIRECT, QUASI or NO_MASK.
      The paper takes DIRECT u QUASI and drops NO_MASK, which is what `identifier_types`
      defaults to.
    - **Which annotator.** Documents carry 1-10 annotators (test split: 22 documents have
      one, 34 have two, 15 have ten). `annotator="first"` takes the first listed -- arbitrary
      but reproducible, and what this repo's earlier results used. `annotator="union"` takes
      every annotator's spans, which raises recall; the paper's Appendix A.17 argues recall
      is the axis that matters, since a missed span falls outside the DP guarantee entirely
      while an over-tagged one only costs utility.

    Not vendored into git: the three splits are 70 MB of json. `dataset/README.md` has the
    fetch command; `dataset/tab_echr/` is where they land.
    """

    name = "tab_echr"
    filename = "tab_echr/echr_test.json"
    SPLITS = ("train", "dev", "test")

    def __init__(
        self,
        split: str = "test",
        path: Optional[Path] = None,
        annotator: str = "first",
        identifier_types: Optional[Sequence[str]] = None,
    ):
        if split not in self.SPLITS:
            raise ValueError(f"unknown split {split!r}, expected one of {self.SPLITS}")
        if annotator not in ("first", "union"):
            raise ValueError(f"annotator must be 'first' or 'union', got {annotator!r}")

        self.split = split
        self.annotator = annotator
        self.identifier_types = set(identifier_types) if identifier_types else set(PRIVATE_IDENTIFIER_TYPES)
        self.filename = f"tab_echr/echr_{split}.json"
        super().__init__(path)

    def __repr__(self) -> str:
        return (f"TabECHR(split={self.split!r}, annotator={self.annotator!r}, "
                f"path={str(self.path)!r})")

    def _records(self) -> List[dict]:
        """One JSON array, not jsonl -- so the base class's line-by-line read does not apply."""
        if not self.path.exists():
            raise FileNotFoundError(
                f"TAB-ECHR {self.split} split not found at {self.path}. It is not vendored "
                "(70 MB); see dataset/README.md for the fetch command, or set "
                "FUSIT_DATASET_DIR to a directory containing tab_echr/."
            )
        with open(self.path) as f:
            return json.load(f)

    def _mentions(self, record: dict) -> List[dict]:
        annotations = record["annotations"]
        if self.annotator == "first":
            return annotations[next(iter(annotations))]["entity_mentions"]
        seen, out = set(), []
        for ann in annotations.values():
            for m in ann["entity_mentions"]:
                key = (m["start_offset"], m["end_offset"], m["entity_type"])
                if key not in seen:
                    seen.add(key)
                    out.append(m)
        return out

    def load(self) -> List[TabDocument]:
        docs = []
        for r in self._records():
            spans = [
                Span(m["start_offset"], m["end_offset"], m["entity_type"], m["span_text"])
                for m in self._mentions(r)
                if m["identifier_type"] in self.identifier_types
            ]
            spans.sort(key=lambda s: (s.start, s.end))
            docs.append(TabDocument(username=r["doc_id"], text=r["text"], spans=spans))
        return docs

    def select(
        self,
        n: Optional[int] = None,
        seed: int = 0,
        require_types: Optional[Sequence[str]] = None,
    ) -> List[TabDocument]:
        """As `Dataset.select`, plus `require_types`: keep only documents carrying at least
        one span of *each* listed type. Pass `ATTACK_ENTITY_TYPES` to guarantee every
        document supports the paper's PERSON/CODE/DATETIME attack scope."""
        docs = self.items()
        if require_types:
            required = set(require_types)
            docs = [d for d in docs if required.issubset(set(d.entity_types()))]
        if n is not None and n < len(docs):
            return random.Random(seed).sample(docs, n)
        return list(docs)

    def stats(self) -> Dict:
        from collections import Counter

        docs = self.items()
        total = sum(len(d.text) for d in docs)
        private = sum(s.end - s.start for d in docs for s in d.spans)
        counts = Counter(s.entity_type for d in docs for s in d.spans)
        return {
            "num_items": len(docs),
            "total_chars": total,
            "avg_chars": total / len(docs) if docs else 0,
            "private_chars": private,
            "private_pct": 100 * private / total if total else 0,
            "num_entities": sum(counts.values()),
            "avg_entities": sum(counts.values()) / len(docs) if docs else 0,
            "entity_type_counts": dict(counts),
        }


_REGISTRY: Dict[str, Type[Dataset]] = {cls.name: cls for cls in (SynthPAI, Synthetic, TabECHR)}
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
