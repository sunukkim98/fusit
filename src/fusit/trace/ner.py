"""
NER(D): the non-adversarial half of `X_priv`, with two interchangeable backends.

Both return the same thing -- char-offset spans into the raw text -- so the choice of
backend never leaks into the rest of the package.

    spacy       `en_core_web_trf`. What this repo ran first. Fast to set up, no extra
                install beyond the spaCy model.
    presidio    Microsoft Presidio driving `dslim/bert-base-NER`, which is the tagger
                DP-Fusion's appendix A.16 selects: "We select the best-performing models
                available within the Presidio suite: BERT-NER (dslim/bert_base_NER)",
                reported there at F1 > 85.4% where spaCy `en_core_web_lg` scored 76.1%.

Presidio is more than the BERT model: alongside the transformer it runs pattern recognizers
for the identifier-shaped PII a CoNLL-trained NER has no class for -- emails, phone numbers,
credit cards, SSNs, IBANs, IPs. That is most of what TAB-ECHR files under CODE.

**Known gap.** BERT-NER's label set is CoNLL's (PER/LOC/ORG/MISC), which has no date class,
so the presidio backend returns *no* DATETIME spans -- even though `DATE_TIME` still appears
in `AnalyzerEngine.get_supported_entities()`, because the recognizer is registered and simply
never fires. Measured here on "She was born on 12 March 1984 and hired in June 2019.":
BERT-NER finds nothing, Presidio's default spaCy engine finds both. That matters for TAB-ECHR,
where DATETIME is one of the three entity types the paper scopes its attack to, so
`presidio_ner_spans(..., with_dates=True)` unions spaCy's DATE entities back in. It is off by
default because it is no longer purely the paper's tagger.
"""

from typing import Dict, List, Optional, Sequence

#: spaCy entity labels treated as private by the `spacy` backend.
NER_LABELS = {"PERSON", "GPE", "LOC", "ORG", "DATE", "NORP", "FAC"}

#: The HF model DP-Fusion A.16 selects.
PRESIDIO_NER_MODEL = "dslim/bert-base-NER"

#: BERT-NER speaks CoNLL; Presidio speaks its own taxonomy.
CONLL_TO_PRESIDIO = {"PER": "PERSON", "LOC": "LOCATION", "ORG": "ORGANIZATION", "MISC": "NRP"}

#: Presidio entity type -> the eight TAB-ECHR categories the paper treats as private
#: (`fusit.utils.ENTITY_TYPES`, which is also how `DEFAULT_BETA_DICT` is keyed). Every
#: identifier-shaped recognizer folds into CODE, matching TAB's own guidelines.
PRESIDIO_TO_ENTITY_TYPE: Dict[str, str] = {
    "PERSON": "PERSON",
    "LOCATION": "LOC",
    "ORGANIZATION": "ORG",
    "NRP": "DEM",
    "DATE_TIME": "DATETIME",
    "CREDIT_CARD": "CODE",
    "CRYPTO": "CODE",
    "EMAIL_ADDRESS": "CODE",
    "IBAN_CODE": "CODE",
    "IP_ADDRESS": "CODE",
    "MAC_ADDRESS": "CODE",
    "MEDICAL_LICENSE": "CODE",
    "PHONE_NUMBER": "CODE",
    "UK_NHS": "CODE",
    "URL": "CODE",
    "US_BANK_NUMBER": "CODE",
    "US_DRIVER_LICENSE": "CODE",
    "US_ITIN": "CODE",
    "US_PASSPORT": "CODE",
    "US_SSN": "CODE",
}

_SPACY_NLP = None
_ANALYZERS: dict = {}


# ---------------------------------------------------------------------------
# spaCy backend
# ---------------------------------------------------------------------------

def _get_nlp(model: str = "en_core_web_trf"):
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy

        _SPACY_NLP = spacy.load(model)
    return _SPACY_NLP


def spacy_ner_spans(text: str) -> List[List[int]]:
    """Entities from `en_core_web_trf` whose label is in `NER_LABELS`."""
    doc = _get_nlp()(text)
    return [[e.start_char, e.end_char] for e in doc.ents if e.label_ in NER_LABELS]


# ---------------------------------------------------------------------------
# Presidio + BERT-NER backend
# ---------------------------------------------------------------------------

def _get_analyzer(model: str = PRESIDIO_NER_MODEL):
    """Build (once per model) a Presidio analyzer whose NLP engine is the BERT tagger.

    spaCy is still loaded, but only as the tokenizer the transformer's word pieces are
    aligned back onto -- `en_core_web_sm` is enough and keeps this off the GPU path.
    """
    if model not in _ANALYZERS:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider
        except ImportError as e:
            raise ImportError(
                'The presidio backend needs: pip install "presidio-analyzer[transformers]" '
                "&& python -m spacy download en_core_web_sm"
            ) from e

        conf = {
            "nlp_engine_name": "transformers",
            "models": [{"lang_code": "en",
                        "model_name": {"spacy": "en_core_web_sm", "transformers": model}}],
            "ner_model_configuration": {
                "model_to_presidio_entity_mapping": dict(CONLL_TO_PRESIDIO),
                "low_confidence_score_multiplier": 0.4,
                "low_score_entity_names": [],
                "labels_to_ignore": ["O"],
            },
        }
        _ANALYZERS[model] = AnalyzerEngine(
            nlp_engine=NlpEngineProvider(nlp_configuration=conf).create_engine(),
            supported_languages=["en"],
        )
    return _ANALYZERS[model]


def presidio_entities(
    text: str,
    entities: Optional[Sequence[str]] = None,
    score_threshold: float = 0.0,
    model: str = PRESIDIO_NER_MODEL,
    with_dates: bool = False,
) -> List[dict]:
    """Every detection, typed -- `{start, end, presidio_type, entity_type, score, text}`.

    `entity_type` is the TAB-ECHR category, so the result can be grouped straight into
    DP-Fusion's per-type privacy groups. Use this when the types matter; `presidio_ner_spans`
    when only the offsets do.
    """
    results = _get_analyzer(model).analyze(
        text=text, language="en",
        entities=list(entities) if entities else None,
        score_threshold=score_threshold,
    )
    out = [
        {
            "start": r.start,
            "end": r.end,
            "presidio_type": r.entity_type,
            "entity_type": PRESIDIO_TO_ENTITY_TYPE.get(r.entity_type, "MISC"),
            "score": float(r.score),
            "text": text[r.start:r.end],
        }
        for r in results
    ]

    if with_dates:
        # BERT-NER has no date class; borrow spaCy's. See the module docstring.
        doc = _get_nlp()(text)
        out += [
            {"start": e.start_char, "end": e.end_char, "presidio_type": "DATE_TIME",
             "entity_type": "DATETIME", "score": 1.0, "text": e.text}
            for e in doc.ents if e.label_ == "DATE"
        ]

    return sorted(out, key=lambda d: (d["start"], d["end"]))


def presidio_ner_spans(
    text: str,
    entities: Optional[Sequence[str]] = None,
    score_threshold: float = 0.0,
    model: str = PRESIDIO_NER_MODEL,
    with_dates: bool = False,
) -> List[List[int]]:
    """Offsets of everything Presidio + BERT-NER flagged."""
    return [[d["start"], d["end"]]
            for d in presidio_entities(text, entities, score_threshold, model, with_dates)]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

BACKENDS = ("spacy", "presidio")


def ner_spans(text: str, backend: str = "spacy", **kwargs) -> List[List[int]]:
    """NER(D) spans from `backend`.

    Defaults to `spacy` so existing results stay reproducible; pass `"presidio"` for the
    tagger DP-Fusion's A.16 selects. Extra kwargs go to the backend (`score_threshold`,
    `with_dates`, `entities`, `model` for presidio).
    """
    if backend == "spacy":
        if kwargs:
            raise TypeError(f"the spacy backend takes no options, got {sorted(kwargs)}")
        return spacy_ner_spans(text)
    if backend == "presidio":
        return presidio_ner_spans(text, **kwargs)
    raise ValueError(f"unknown backend {backend!r}, expected one of {BACKENDS}")


__all__ = [
    "BACKENDS", "CONLL_TO_PRESIDIO", "NER_LABELS", "PRESIDIO_NER_MODEL",
    "PRESIDIO_TO_ENTITY_TYPE", "ner_spans", "presidio_entities", "presidio_ner_spans",
    "spacy_ner_spans",
]
