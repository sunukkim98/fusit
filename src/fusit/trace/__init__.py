"""
TRACE: attribute-inference attack machinery, and the cue tagger built on it.

DP-Fusion's original tagger marks named entities. That is the right target when the secret is
written down -- a name, a date, an account number -- and the wrong one when the secret is
*inferable*: an author never states their occupation, yet "juggling visuals, concepts and
timelines" gives it away, and there is no entity there to tag.

This package implements the cue tagger from *Bridging TRACE and DP-Fusion for Implicit PII*,
which widens the target to `X_priv = NER(D) u V_cot u V_att` -- entities, plus the text an
adversarial model quotes when justifying its own guess, plus the words it attends to while
making that guess. Prompts and scoring follow TRACE-RPS (Apache-2.0,
https://arxiv.org/abs/2602.11528); see `prompts` for what is copied verbatim and why.

    >>> from fusit.trace import CueTagger
    >>> tagger = CueTagger(model, tokenizer, attributes=["occupation"])
    >>> tagger.extract_spans(document)            # char offsets -- the lossless form
    >>> tagger.explain(document)                  # {source: spans}, for the ablation
    >>> tagger.extract_private_phrases(document)  # the fusit.dp_fusion.Tagger contract

`CueTagger` satisfies the tagger contract `fusit.dp_fusion` already expects, so it can be
handed to `DPFusion(tagger=...)` unchanged -- see `fusit.trace.tagger` for the one caveat
that makes `extract_spans` the better call where a caller can use it.

Modules:
    attributes  the attribute vocabulary and its prompt-facing labels/questions
    prompts     TRACE-RPS prompt text, verbatim
    chat        one portable chat call (template fallback, context budget)
    parsing     reading guesses and evidence quotes back out of free text
    spans       span algebra, plus the controls a cue-quality claim needs
    ner         NER(D): spaCy, or Presidio + BERT-NER as in DP-Fusion's appendix A.16
    signals     the three sources: ner_spans, infer_and_chain, attention_spans
    tagger      CueTagger
"""

from fusit.trace.attributes import (
    ATTRIBUTES,
    ATTRIBUTE_LABEL,
    ATTRIBUTE_OPTIONS,
    ATTRIBUTE_QUESTION,
    label_of,
    question_of,
)
from fusit.trace.chat import chat, format_chat
from fusit.trace.prompts import (
    ADVERSARIAL_INFERENCE_QUERY_PROMPT_TEMPLATE,
    ADVERSARIAL_INFERENCE_SYSTEM_PROMPT,
    FUNCTIONAL_WORDS,
    PRIVACY_LEAKAGE_CHAIN_PROMPT_TEMPLATE,
    PRIVACY_LEAKAGE_CHAIN_SYSTEM_PROMPT,
)
from fusit.trace.parsing import (
    extract_evidence_quotes,
    parse_inference_response,
    split_guesses,
)
from fusit.trace.ner import (
    BACKENDS,
    NER_LABELS,
    PRESIDIO_NER_MODEL,
    PRESIDIO_TO_ENTITY_TYPE,
    ner_spans,
    presidio_entities,
    presidio_ner_spans,
    spacy_ner_spans,
)
from fusit.trace.signals import (
    attention_spans,
    guess_attribute,
    infer_and_chain,
    oracle_gt_spans,
)
from fusit.trace.spans import (
    content_word_spans,
    coverage,
    is_functional_word,
    merge_spans,
    oracle_all_content,
    random_spans_matched,
    redact,
)
from fusit.trace.tagger import SOURCES, CueTagger, build_x_priv

__all__ = [
    "ADVERSARIAL_INFERENCE_QUERY_PROMPT_TEMPLATE", "ADVERSARIAL_INFERENCE_SYSTEM_PROMPT",
    "FUNCTIONAL_WORDS", "PRIVACY_LEAKAGE_CHAIN_PROMPT_TEMPLATE",
    "PRIVACY_LEAKAGE_CHAIN_SYSTEM_PROMPT",
    "ATTRIBUTES", "ATTRIBUTE_LABEL", "ATTRIBUTE_OPTIONS", "ATTRIBUTE_QUESTION",
    "BACKENDS", "CueTagger", "NER_LABELS", "PRESIDIO_NER_MODEL",
    "PRESIDIO_TO_ENTITY_TYPE", "SOURCES", "build_x_priv",
    "attention_spans", "chat", "content_word_spans", "coverage", "extract_evidence_quotes",
    "format_chat", "guess_attribute", "infer_and_chain", "is_functional_word", "label_of",
    "merge_spans", "ner_spans", "oracle_all_content", "oracle_gt_spans",
    "parse_inference_response", "presidio_entities", "presidio_ner_spans",
    "question_of", "random_spans_matched", "redact", "spacy_ner_spans",
    "split_guesses",
]
