# dataset/

Vendored ground-truth corpora. Data only — the code that reads them is
[`fusit.dataset`](../src/fusit/dataset.py).

| file | items | md5 | size |
| --- | --- | --- | --- |
| `synthpai.jsonl` | 298 profiles | `2c926d8cc0e12eb87593e099c7cfc116` | 1.7 MB |
| `synthetic_dataset.jsonl` | 525 comments | `1ccf51e3b2f49e46546d2ee8ed5b3bfd` | 1.2 MB |

## Provenance

Both files are copied verbatim from `data/synthetic/` of
[TRACE-RPS](https://github.com/sunukkim98/TRACE-RPS) at commit `86449226`
(2026-04-06), which carries the SRI Lab / ETH Zurich MIT-licensed SynthPAI material.
Copied 2026-09-16; unmodified, so the md5s above still match upstream.

TAB-ECHR is **not** vendored here (88 MB of json). It stays a separate checkout of
[text-anonymization-benchmark](https://github.com/NorskRegnesentral/text-anonymization-benchmark),
found by default as a sibling of this repo and loaded by `repro/data.py`
(override with `FUSIT_TAB_DIR`).

## Reading them

```python
from fusit.dataset import DATASETS, get_dataset, load_dataset

ds = get_dataset("synthpai")        # memoized; parses the jsonl once per process
items = ds.select(n=50, seed=0)     # fixed-seed subsample, or all of it when n is None
items = load_dataset("synthetic")   # one-shot equivalent
```

Every item has `.username`, `.text` and `.relevant_pii` (attribute -> ground-truth value),
so code that handles both datasets need not branch on the name. They differ in unit and in
how ground truth is filtered — SynthPAI items are *profiles* with many attributes (kept when
`hardness >= 1 and certainty >= 1` on `reviews.human`, mirroring TRACE-RPS's
`Profile.get_relevant_pii`), synthetic items are single *comments* with exactly one target
attribute plus a `.hardness`, where that filter is vacuous by construction. See each class's
docstring for the details.

Corpus statistics:

```
python -m fusit.dataset              # both
python -m fusit.dataset synthpai     # one
```

## Paths

`fusit.dataset.DATASET_DIR` resolves to this directory relative to the package source, so an
editable install (`pip install -e .`) needs no configuration. These jsonl sit outside `src/`
and are therefore **not** shipped in a built wheel — for a non-editable install, point
`FUSIT_DATASET_DIR` at a directory holding the two files. A missing file raises with that
instruction rather than a bare `FileNotFoundError`.
