"""
Prompt construction for DP-Fusion generation.

The private and public contexts must tokenize to the SAME length for the mechanism's
length-leakage guarantee to hold, so both are built from this one template and differ only
in which spans have been swapped for the placeholder.

The template pre-fills the start of the assistant's reply ("Sure. Here is the paraphrased
document...") and then **leaves the turn open**, which is how the paper prints it in
Appendix A.8. Closing it instead hands the model a finished conversation, and it answers by
emitting EOS immediately: on TAB-ECHR with Qwen2.5-0.5B-Instruct that produced an empty
paraphrase for 10 documents out of 10. `open_turn=False` restores the closed form for anyone
who needs the old behaviour back.

Two templates live here:

    official_prompt              the official DP-Fusion-DPI template, verbatim
                                 (DP-FUSION_Defense.py::format_prompt_new_template): instructions
                                 in the system turn, hard-coded ChatML. Every DP-Fusion run of the
                                 verifier experiments (implicit table, explicit Table 1, PPL) uses it.
    format_prompt_new_template   fusit's own rendering through the tokenizer's chat template.
                                 DIFFERS from the official one (decision D9, 2026-10-03): the
                                 instructions move into the user turn, Qwen's default system prompt
                                 is added, the typo "as a natural as" is fixed and line breaks change.
                                 Kept for fusit.main / fusit.perplexity, which were run with it.
"""

_SENTINEL = "<<<FUSIT_ASSISTANT_PREFILL>>>"

#: DP-FUSION_Defense.py::format_prompt_new_template, verbatim (typo "as a natural as" included).
#: {privacy} is empty for DP-Fusion; the No-DPI baselines insert " of this for ensuring privacy"
#: (outputs_qwen_public.json).
OFFICIAL_TEMPLATE = (
    "<|im_start|>system\n"
    "You are given a passage that may contain placeholders (underscores) or incomplete data. "
    "Your job is to produce a natural paraphrase{privacy}. Do not use any underscores or "
    "placeholders in your output. If data is missing, just omit it or paraphrase gracefully. "
    "Do not output anything except the paraphrase. Make sure retain all information from the "
    "source document.<|im_end|>\n"
    "<|im_start|>user\n"
    "Document:\n\n{doc}\n"
    "Paraphrase the above text. Whenever a placeholder i.e {ph} exists, you must completely "
    "ignore that information, as {ph} indicates redacted text. To ensure the generated text is "
    "as a natural as possible, you must never output the {ph} themselves. <|im_end|>\n"
    "<|im_start|>assistant\n"
    "Sure. Here is the paraphrased document without underscores or placeholders:"
)
NODPI_PRIVACY = " of this for ensuring privacy"


def official_prompt(tokenizer, doc: str, placeholder: str = "_") -> str:
    """The official DP-Fusion prompt. `tokenizer` is unused (the template is hard-coded ChatML);
    it is kept so this has the same (tokenizer, text) signature as the other prompt builders."""
    return OFFICIAL_TEMPLATE.format(privacy="", doc=doc, ph=placeholder)


def official_nodpi_prompt(tokenizer, doc: str, placeholder: str = "_") -> str:
    """The official template with the No-DPI baselines' privacy instruction."""
    return OFFICIAL_TEMPLATE.format(privacy=NODPI_PRIVACY, doc=doc, ph=placeholder)


def format_prompt_new_template(
    tokenizer, private_doc: str, placeholder: str, open_turn: bool = True
) -> str:
    """
    Create a prompt using the HuggingFace chat template.

    Args:
        tokenizer: HuggingFace tokenizer
        private_doc: Document to paraphrase (may contain placeholders)
        placeholder: Placeholder character (e.g., "_")
        open_turn: leave the assistant turn open so the model continues the pre-filled
            sentence, as in the paper's Appendix A.8. Set False to close it with the
            template's end-of-turn token, which stops generation before it starts.

    Returns:
        Formatted prompt for the model
    """
    system_text = (
        "You are given a passage that may contain placeholders (underscores) or incomplete data. "
        "Your job is to produce a natural paraphrase. "
        "Do not use any underscores or placeholders in your output. "
        "If data is missing, just omit it or paraphrase gracefully. "
        "Do not output anything except the paraphrase. "
        "Make sure retain all information from the source document."
    )

    user_task = (
        f"Document:\n\n{private_doc}\n\n"
        f"Paraphrase the above text. Whenever a placeholder i.e {placeholder} exists, you must completely ignore that information, "
        f"as {placeholder} indicates redacted text. To ensure the generated text is as natural as possible, "
        f"you must never output the {placeholder} themselves."
    )

    messages = [
        {"role": "user", "content": f"{system_text}\n\n{user_task}"},
        {"role": "assistant", "content": "Sure. Here is the paraphrased document without underscores or placeholders:"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    if open_turn:
        # Cut off whatever the chat template appends AFTER the assistant's content, so the
        # pre-filled sentence stays open. Measure that suffix rather than guessing at it:
        # templates differ (Qwen closes with <|im_end|>, Llama-3.1 closes with <|eot_id|> and
        # then adds a fresh assistant header even with add_generation_prompt=False), and a
        # hardcoded marker list silently produces a wrong prompt on the next model.
        probe_messages = [dict(m) for m in messages]
        probe_messages[-1]["content"] = _SENTINEL
        probe = tokenizer.apply_chat_template(
            probe_messages, tokenize=False, add_generation_prompt=False
        )
        if probe.count(_SENTINEL) != 1:
            raise ValueError(
                f"cannot locate the assistant turn in this chat template "
                f"({probe.count(_SENTINEL)} sentinel matches); pass open_turn=False."
            )
        suffix = probe[probe.index(_SENTINEL) + len(_SENTINEL):]
        if suffix and not prompt.endswith(suffix):
            raise ValueError(
                "chat template rendered inconsistently between probe and prompt; "
                "pass open_turn=False."
            )
        return prompt[: len(prompt) - len(suffix)] if suffix else prompt

    return prompt
