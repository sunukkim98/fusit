"""
`CueTagger` -- the package's entry point, and the seam with DP-Fusion.

`fusit.dp_fusion` already accepts a tagger object and asks it one question:

    tagger.extract_private_phrases(document) -> list[str]

`CueTagger` answers it, so it drops straight into `DPFusion(tagger=...)` in place of the
Document Privacy API client, with no change to `fusit.dp_fusion.core`:

    >>> from fusit import DPFusion
    >>> from fusit.trace import CueTagger
    >>> tagger = CueTagger(model, tokenizer, attributes=["occupation"])
    >>> dpf = DPFusion(model=model, tokenizer=tokenizer, tagger=tagger)
    >>> dpf.add_message("user", profile_text, is_private=True)
    >>> dpf.run_tagger()

**Prefer `extract_spans` where you can.** Phrases are the lossy form. `run_tagger` feeds them
back through `find_phrase_offsets`, which re-matches each phrase *everywhere* it occurs in the
full prompt -- so a one-word attention cue like "oh" redacts every "oh" in the document, and a
phrase that happens to appear in the system prompt is redacted there too. `extract_spans`
returns the offsets the signals actually selected, which is what a caller holding the raw
document should use; `repro/`'s pipelines already do.

Sources are selectable so the ablation the experiment plan asks for (NER alone vs NER+cues) is
a constructor argument rather than a separate code path.
"""

from typing import Dict, List, Optional, Sequence

from fusit.trace.attributes import ATTRIBUTES, question_of
from fusit.trace.ner import BACKENDS, ner_spans
from fusit.trace.signals import attention_spans, infer_and_chain
from fusit.trace.spans import coverage, merge_spans

#: Signal sources `CueTagger` knows how to run.
SOURCES = ("ner", "cot", "att")


class CueTagger:
    """Union of the requested signal sources over the requested attributes.

    Args:
        model: a HuggingFace CausalLM, used for the `cot` and `att` sources. May be None
            when `sources` is `{"ner"}` only.
        tokenizer: its tokenizer.
        attributes: which attributes to hunt for. Defaults to all of `ATTRIBUTES`; passing
            the handful you actually score is much faster, since `cot` costs two generations
            per attribute.
        sources: subset of `SOURCES`. Default is the paper's full NER u V_cot u V_att.
        k: how many top-attention words `att` keeps per attribute.
        ner_backend: "spacy" (default, what this repo ran first) or "presidio" for the
            Presidio + BERT-NER tagger DP-Fusion's appendix A.16 selects. See
            `fusit.trace.ner`.
        ner_options: extra keyword arguments for the backend, e.g.
            `{"score_threshold": 0.4, "with_dates": True}` for presidio.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        attributes: Optional[Sequence[str]] = None,
        sources: Sequence[str] = SOURCES,
        k: int = 10,
        ner_backend: str = "spacy",
        ner_options: Optional[Dict] = None,
    ):
        unknown = set(sources) - set(SOURCES)
        if unknown:
            raise ValueError(f"unknown sources {sorted(unknown)}, expected a subset of {SOURCES}")
        if not sources:
            raise ValueError("at least one source is required")

        self.attributes = list(attributes) if attributes is not None else list(ATTRIBUTES)
        for attr in self.attributes:
            if attr not in ATTRIBUTES:
                raise ValueError(f"unknown attribute {attr!r}, expected one of {ATTRIBUTES}")

        self.sources = set(sources)
        if self.sources - {"ner"} and model is None:
            raise ValueError(f"sources {sorted(self.sources)} need a model; only 'ner' runs without one")

        if ner_backend not in BACKENDS:
            raise ValueError(f"unknown ner_backend {ner_backend!r}, expected one of {BACKENDS}")

        self.model = model
        self.tokenizer = tokenizer
        self.k = k
        self.ner_backend = ner_backend
        self.ner_options = dict(ner_options or {})

    def __repr__(self) -> str:
        return (f"CueTagger(sources={sorted(self.sources)}, "
                f"attributes={self.attributes}, k={self.k}, "
                f"ner_backend={self.ner_backend!r})")

    # -- the offset interface (preferred) --------------------------------------

    def extract_spans(
        self, document: str, attributes: Optional[Sequence[str]] = None
    ) -> List[List[int]]:
        """Merged char-offset spans of everything the enabled sources flagged."""
        return merge_spans(
            [s for spans in self.explain(document, attributes).values() for s in spans]
        )

    def explain(
        self, document: str, attributes: Optional[Sequence[str]] = None
    ) -> Dict[str, List[List[int]]]:
        """`extract_spans` without the union: `{source: spans}`.

        Which source contributed what is the whole subject of the ablation, and it is gone
        once the spans are merged -- so the breakdown is produced here rather than
        reconstructed by running each source again.
        """
        attrs = list(attributes) if attributes is not None else self.attributes
        out: Dict[str, List[List[int]]] = {s: [] for s in sorted(self.sources)}

        if "ner" in self.sources:
            out["ner"] = ner_spans(document, backend=self.ner_backend, **self.ner_options)

        for attr in attrs:
            if "att" in self.sources:
                out["att"] += attention_spans(
                    document, question_of(attr), self.model, self.tokenizer, k=self.k
                )
            if "cot" in self.sources:
                out["cot"] += infer_and_chain(document, attr, self.model, self.tokenizer)["evidence_spans"]

        return {src: merge_spans(spans) for src, spans in out.items()}

    # -- the fusit.dp_fusion tagger interface ----------------------------------

    def extract_private_phrases(self, document: str) -> List[str]:
        """The substrings `extract_spans` selected.

        Present so `CueTagger` satisfies the same contract as
        `fusit.dp_fusion.tagger.Tagger` and can be handed to `DPFusion`. Lossy -- see the
        module docstring -- so reach for `extract_spans` when the caller has the offsets.
        """
        return [document[s:e] for s, e in self.extract_spans(document)]

    # -- reporting -------------------------------------------------------------

    def coverage(self, document: str, attributes: Optional[Sequence[str]] = None) -> float:
        """Fraction of the document `extract_spans` would redact."""
        return coverage(document, self.extract_spans(document, attributes))


def build_x_priv(
    text: str,
    attributes: Sequence[str],
    model=None,
    tokenizer=None,
    k: int = 10,
    sources: Optional[Sequence[str]] = None,
    ner_backend: str = "spacy",
    ner_options: Optional[Dict] = None,
) -> List[List[int]]:
    """Functional form of `CueTagger(...).extract_spans(text)`.

    The tagger holds no state between documents, so building one per call costs nothing; this
    exists because sweeping over source subsets reads better as an argument than as a
    constructor.
    """
    tagger = CueTagger(
        model, tokenizer, attributes=attributes,
        sources=sorted(sources) if sources else SOURCES, k=k,
        ner_backend=ner_backend, ner_options=ner_options,
    )
    return tagger.extract_spans(text)


__all__ = ["SOURCES", "CueTagger", "build_x_priv"]
