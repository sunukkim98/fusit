"""
Helpers shared across the package.

Nothing here is specific to a single method: the entity taxonomy and placeholder used when
marking text as private, plus the two span/token primitives that operate on them. The
DP-Fusion algorithm itself lives in `fusit.dp_fusion`.
"""

from bisect import bisect_right
from typing import List

# Entity types available
ENTITY_TYPES = [
    "PERSON", "CODE", "LOC", "ORG", "DEM",
    "DATETIME", "QUANTITY", "MISC"
]

# Default placeholder token for redaction
PLACEHOLDER_TOKEN = "_"


def replace_sequences_with_placeholder_fast(
    text: str,
    word_offsets: List[List[int]],
    placeholder: str,
    tokenizer
) -> List[int]:
    """
    Replace tokens falling within provided word offset ranges with placeholder tokens.

    Args:
        text: Original text string
        word_offsets: List of [start_char, end_char] offsets for words to replace
        placeholder: Placeholder token to use (e.g., "_")
        tokenizer: Tokenizer that returns 'input_ids' and 'offset_mapping'

    Returns:
        Token IDs with specified words replaced by placeholder token ID
    """
    placeholder_id = tokenizer.convert_tokens_to_ids(placeholder)

    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = encoded['input_ids']
    offsets = encoded['offset_mapping']

    word_offsets = sorted(word_offsets, key=lambda x: x[0])
    starts = [wo[0] for wo in word_offsets]
    ends = [wo[1] for wo in word_offsets]

    for i, (t_start, t_end) in enumerate(offsets):
        if t_start == t_end:
            continue

        idx = bisect_right(starts, t_end)

        while idx > 0:
            idx -= 1
            w_start, w_end = starts[idx], ends[idx]

            if w_end > t_start and w_start < t_end:
                input_ids[i] = placeholder_id
                break

    return input_ids


def find_phrase_offsets(text: str, phrases: List[str]) -> List[List[int]]:
    """
    Find all occurrences of phrases in text and return [start, end] offsets.

    Args:
        text: The full text to search in
        phrases: List of phrases to find

    Returns:
        List of [start_char, end_char] offsets for all phrase occurrences
    """
    offsets = []
    for phrase in phrases:
        start = 0
        while True:
            idx = text.find(phrase, start)
            if idx == -1:
                break
            offsets.append([idx, idx + len(phrase)])
            start = idx + 1
    return offsets
