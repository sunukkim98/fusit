"""
Prompt construction for DP-Fusion generation.

The private and public contexts must tokenize to the SAME length for the mechanism's
length-leakage guarantee to hold, so both are built from this one template and differ only
in which spans have been swapped for the placeholder.
"""

def format_prompt_new_template(tokenizer, private_doc: str, placeholder: str) -> str:
    """
    Create a prompt using the HuggingFace chat template.

    Args:
        tokenizer: HuggingFace tokenizer
        private_doc: Document to paraphrase (may contain placeholders)
        placeholder: Placeholder character (e.g., "_")

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

    return prompt
