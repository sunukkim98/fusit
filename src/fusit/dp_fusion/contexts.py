"""
Building the contexts DP-Fusion generates from.

Algorithm 1 needs one token sequence per privacy group plus a public one, and they must all
have **identical token length**. That is not a detail: the paper pads each redacted span with
the same number of placeholder tokens precisely so that `X_pub` and `X_pub u X_i` line up,
"to prevent length-based leakage". A shorter public context would leak how much was redacted
before a single token is generated.

Tokenizing each context separately does not give you that. Measured on TAB-ECHR document
001-91088, the private prompt came to 522 tokens and the public one to 469 -- so the two runs
are not even comparable, let alone length-matched. The fix, and what these builders do, is to
tokenize the full prompt **once** and swap only the token ids that fall inside a redacted
span, via `fusit.utils.replace_sequences_with_placeholder_fast`.

    build_contexts        the paper's m-group setup (Figure 2): PUBLIC reveals nothing, and
                          group i reveals only its own entity type
    build_aligned_tokens  the two-group case: one PUBLIC, one PRIVATE
"""

from typing import Dict, List, Optional, Sequence

from fusit.dp_fusion.prompting import format_prompt_new_template
from fusit.utils import replace_sequences_with_placeholder_fast

__all__ = ["build_aligned_tokens", "build_contexts", "locate_document"]


def locate_document(prompt: str, text: str) -> int:
    """Offset of `text` inside `prompt`, so document-relative spans can be shifted into it.

    Requires a unique match: if the document appears twice the offsets are ambiguous and the
    redaction would silently apply to the wrong copy.
    """
    if prompt.count(text) != 1:
        raise ValueError(
            f"document text occurs {prompt.count(text)} times in the prompt, expected once; "
            "span offsets cannot be located unambiguously."
        )
    return prompt.find(text)


def _redact(prompt: str, offsets: Sequence[Sequence[int]], placeholder: str, tokenizer) -> List[int]:
    return replace_sequences_with_placeholder_fast(
        prompt, [list(o) for o in offsets], placeholder, tokenizer
    )


def build_contexts(
    tokenizer,
    text: str,
    spans,
    entity_types: Optional[Sequence[str]] = None,
    placeholder: str = "_",
) -> Dict[str, List[int]]:
    """`{"PUBLIC": tokens, "<TYPE>": tokens, ...}`, all of equal length.

    Args:
        text: the document to paraphrase.
        spans: objects with `.start`, `.end` and `.entity_type` -- `fusit.dataset.Span` fits.
        entity_types: which types get their **own group**. Defaults to every type present.
            Types with no spans are skipped: a group with nothing to reveal costs a forward
            pass per token and changes nothing.

    **Every span is redacted from PUBLIC, whether or not its type is a group.** The paper
    keeps these separate and it is easy to conflate them: "all entity groups are treated as
    private in DP-Fusion, we focus the evaluation of our attacks only against the PERSON,
    CODE, and DATETIME groups" (Section 5.1). Narrowing `entity_types` narrows what gets its
    own epsilon, not what stays secret -- a span of some other type is simply never revealed
    in any context, which is strictly stronger than giving it a group. Revealing it instead
    would hand the model private text with no accounting at all, and it would come out in the
    paraphrase: on TAB-ECHR 001-79071 with groups PERSON/CODE/DATETIME, the LOC span
    "Sundbyberg" reached the output verbatim.

    Raises if the contexts do not come out the same length, since that is the one property
    the caller cannot check later from the token ids alone.
    """
    spans = list(spans)
    present = [t for t in (entity_types or sorted({s.entity_type for s in spans}))
               if any(s.entity_type == t for s in spans)]
    grouped = set(present)

    prompt = format_prompt_new_template(tokenizer, text, placeholder)
    start = locate_document(prompt, text)

    def offsets(ss):
        return [[start + s.start, start + s.end] for s in ss]

    groups = {"PUBLIC": _redact(prompt, offsets(spans), placeholder, tokenizer)}
    for t in present:
        # group t reveals ONLY type t; everything else -- including spans of ungrouped types
        # -- stays redacted
        groups[t] = _redact(prompt, offsets([s for s in spans if s.entity_type != t]),
                            placeholder, tokenizer)

    lengths = {g: len(v) for g, v in groups.items()}
    if len(set(lengths.values())) != 1:
        raise AssertionError(f"context token lengths differ, breaking the guarantee: {lengths}")
    return groups


def build_aligned_tokens(
    tokenizer,
    text: str,
    spans,
    entity_types: Optional[Sequence[str]] = None,
    placeholder: str = "_",
):
    """The two-group case: `(private_tokens, public_tokens)`, guaranteed equal length.

    `private_tokens` is the unredacted prompt; `public_tokens` has every span of
    `entity_types` (default: all of them) swapped for the placeholder.
    """
    spans = list(spans)
    if entity_types is not None:
        allowed = set(entity_types)
        spans = [s for s in spans if s.entity_type in allowed]

    prompt = format_prompt_new_template(tokenizer, text, placeholder)
    start = locate_document(prompt, text)

    private = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    public = _redact(prompt, [[start + s.start, start + s.end] for s in spans],
                     placeholder, tokenizer)
    if len(private) != len(public):
        raise AssertionError(
            f"private/public token counts differ ({len(private)} vs {len(public)}), "
            "breaking the length-leakage guarantee."
        )
    return private, public
