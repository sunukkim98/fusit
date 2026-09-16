# dataset/

Vendored ground-truth corpora. Data only — the code that reads them is
[`fusit.dataset`](../src/fusit/dataset.py).

| file | items | md5 | size | tracked |
| --- | --- | --- | --- | --- |
| `synthpai.jsonl` | 298 profiles | `2c926d8cc0e12eb87593e099c7cfc116` | 1.7 MB | yes |
| `synthetic_dataset.jsonl` | 525 comments | `1ccf51e3b2f49e46546d2ee8ed5b3bfd` | 1.2 MB | yes |
| `tab_echr/echr_train.json` | 1,014 documents | `eec25afa150b3e29acc562f3e0fcf377` | 47.5 MB | no |
| `tab_echr/echr_dev.json` | 127 documents | `fdaed831a90048263336c89ee0af8c55` | 12.6 MB | no |
| `tab_echr/echr_test.json` | 127 documents | `988de4f287a96233b73db73369e41005` | 13.1 MB | no |

## Provenance

### SynthPAI and synthetic

Both jsonl are copied verbatim from `data/synthetic/` of
[TRACE-RPS](https://github.com/sunukkim98/TRACE-RPS) at commit `86449226`
(2026-04-06), which carries the SRI Lab / ETH Zurich MIT-licensed SynthPAI material.
Copied 2026-09-16; unmodified, so the md5s above still match upstream.

## TAB-ECHR

DP-Fusion's own evaluation corpus (paper Section 5.1): European Court of Human Rights
judgments from the
[Text Anonymization Benchmark](https://github.com/NorskRegnesentral/text-anonymization-benchmark)
(Pilán et al., 2022), with private information hand-annotated in eight categories. MIT
licensed; `tab_echr/LICENSE.txt` and `tab_echr/guidelines.md` are kept here alongside it.

**The three splits are gitignored** — 70 MB of json is too much to carry in the repo — so
this is the one corpus you have to fetch. Two ways:

**1. Our Drive copy** —
<https://drive.google.com/file/d/14oTnnNoSlJy6z9__BfpzTebYTu2-ePwT/view>

Open that link and download, or from a shell:

```bash
pip install gdown
gdown 14oTnnNoSlJy6z9__BfpzTebYTu2-ePwT
```

Then put the three `echr_*.json` inside `dataset/tab_echr/`, extracting first if the download
is an archive.

**2. Or straight from upstream:**

```bash
git clone https://github.com/NorskRegnesentral/text-anonymization-benchmark
cp text-anonymization-benchmark/echr_*.json dataset/tab_echr/
```

Either way the result should be exactly this, which the md5s in the table above confirm:

```
dataset/tab_echr/
├── echr_train.json
├── echr_dev.json
└── echr_test.json

$ md5sum dataset/tab_echr/*.json
```

`FUSIT_DATASET_DIR` relocates the whole `dataset/` directory if the files live elsewhere.

## Reading them

```python
from fusit.dataset import DATASETS, get_dataset, load_dataset

ds = get_dataset("synthpai")        # memoized; parses the jsonl once per process
items = ds.select(n=50, seed=0)     # fixed-seed subsample, or all of it when n is None
items = load_dataset("synthetic")   # one-shot equivalent

tab = get_dataset("tab_echr")       # == TabECHR(split="test")
doc = tab.select(n=1, seed=0)[0]
doc.offsets()                       # [[start, end], ...] of every private span
```

Every item has `.username`, `.text` and `.relevant_pii` (attribute -> ground-truth value),
so code that handles both datasets need not branch on the name. They differ in unit and in
how ground truth is filtered — SynthPAI items are *profiles* with many attributes (kept when
`hardness >= 1 and certainty >= 1` on `reviews.human`, mirroring TRACE-RPS's
`Profile.get_relevant_pii`), synthetic items are single *comments* with exactly one target
attribute plus a `.hardness`, where that filter is vacuous by construction. See each class's
docstring for the details.

**TAB-ECHR does not share that contract**, on purpose. It answers a different question —
*which characters are private*, not *which attribute value is inferable* — so a `TabDocument`
carries typed `Span`s instead of `relevant_pii` and is not an `Item`. Forcing it into the same
shape would mean an empty `relevant_pii` on every document, which anything trusting that field
would read as "nothing to protect here".

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
