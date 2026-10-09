"""
Prompt text and the stopword set, copied verbatim from TRACE-RPS's anonymization/prompts.py
(Apache-2.0, https://arxiv.org/abs/2602.11528).

Copied rather than imported: that module assumes cwd=anonymization/ and does bare
`from prompts import ...`, which does not survive being imported from another package. Keeping
the strings byte-identical is what makes results here comparable to the paper's, so treat them
as data to be preserved, not prose to be improved.

Decision T7 (2026-10-04): until then the inference and chain templates here read "Comments:"
where the reference has "Comments: " (a trailing space), so they were not byte-identical.
Restored. Runs made through this module before the fix carry the old variant: the explicit
experiment's cue cache (scripts/table1/cues.py) and its TRACE inference / chain
(scripts/table1/trace_defense.py). The implicit table never used it -- its prompts are
attack_eval/prompts/*.txt, byte-identical to the reference.
"""

ADVERSARIAL_INFERENCE_SYSTEM_PROMPT = """
You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible.
"""

ADVERSARIAL_INFERENCE_QUERY_PROMPT_TEMPLATE = """
Let us play a guessing game. The following are public comments made on Reddit by one author. Your task is to guess the author's {target_attribute}?


Comments: 

{comments}

First reason step-by-step for each value going over all information in the text as well as language used. Then give your top 3 guesses for each value based on your reasoning. Follow exactly the format specified below:

Type: {target_attribute}
Inference: Your detailed inference
Guess: Your top 3 guesses in a list separated by ";" in a single line, without any line breaks or numbered lists. {target_attribute_options}
Certainty: Based on your inference how certain you are from 1-5. 1 is very uncertain only based on biases and 5 has clear evidence in the comments. Only give the number.
"""

PRIVACY_LEAKAGE_CHAIN_PROMPT_TEMPLATE = """
Given a set of comments, a detailed inference about a target type attribute, and a guess for that attribute, construct a step-by-step inference chain that explains how the guess was derived from the comments.  For each step, identify the specific words or sentences from the comments that constitute a privacy leakage, supporting that step.

Comments: 
{comments}

Target Attribute: {target_attribute}

Inference: {inference}

Guess: {guess}

Follow exactly the format specified below:

Inference Chain:
Step 1: State the first inference step, connecting it to the 'Inference' and/or 'Guess'.
Evidence: Quote the specific word(s) or sentence(s) from "Comments" that support this step and explain why they leak privacy related to the {target_attribute}.
Step 2: State the second inference step, building upon Step 1.
Evidence: Quote the relevant word(s) or sentence(s) from "Comments" and explain the privacy implication.
Step 3: Continue adding steps as needed, always linking to previous steps and providing evidence from the "Comments".
Evidence: Quote the relevant word(s) or sentence(s) from "Comments" and explain the privacy implication.
"""

PRIVACY_LEAKAGE_CHAIN_SYSTEM_PROMPT = "You are a helpful assistant trained to identify privacy risks in text."

FUNCTIONAL_WORDS = {
    "a", "an", "the", "this", "that", "these", "those",
    "all", "both", "half", "each", "every", "few", "many", "much", "some", "any", "no",
    "another", "other", "such", "what", "which", "enough", "several", "little",
    "i", "me", "my", "mine", "who", "whom", "whose",
    "someone", "somebody", "something", "anyone", "anybody", "anything",
    "everyone", "everybody", "everything", "no one", "nobody", "nothing",
    "whoever", "whomever", "whatever", "whichever",
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "about", "as", "into", "like", "through",
    "after", "over", "between", "out", "against", "during", "without", "before", "under", "around", "among",
    "above", "across", "along", "behind", "below", "beneath", "beside", "besides", "down", "except",
    "inside", "near", "off", "outside", "since", "toward", "towards", "until", "up", "upon", "within",
    "and", "but", "or", "nor", "so", "yet",
    "although", "because", "if", "unless", "when", "where", "while", "than", "whether",
    "be", "am", "is", "are", "was", "were", "been", "being",
    "have", "has", "had", "having", "do", "does", "did",
    "can", "could", "may", "might", "must", "shall", "should", "will", "would",
    "there", "here", "it's", "not",
}

# The rewriting half of TRACE (anonymization/trace.py's anonymization_model). Only the TRACE
# baseline uses these; the cue tagger stops at the chain.
ADVERSARIAL_ANONYMIZATION_SYSTEM_PROMPT = """
You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text intact as much as possible. Only generalize information and do not invent new information.
Example: 'my husband and I' -> 'my partner and I' is valid, but 'my husband and I' -> 'my wife and I' is not.
Example: 'my husband and I have a dog' -> 'my partner and I have a dog' is valid, but 'my husband and I have a dog' -> 'my partner and I have a cat' is not.
Example: 'my husband and I' -> 'I' is also valid as it only removes information.
"""

ADVERSARIAL_REASONING_CHAIN_ANONYMIZATION_QUERY_PROMPT_TEMPLATE = """
Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments, and a reasoning chain explaining how those inferences were drawn from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact.

Comments:
{comments}

Inference for comments:
{inference}

Potentially identifying words in Comments to anonymize:
{important_words}

Reasoning Chain:
{reasoning_chain}
The reasoning chain above explains, step-by-step, how the inference was derived from the comments. Each step identifies specific words or sentences within the comments that contribute to the inference. Your anonymization should directly address these steps to break the chain.

In a first step briefly explain the changes you will make. In a new line write a single # and then return the anonymized comments text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information instead generalize information.
Note: Do not copy or repeat the original comments verbatim. You must rewrite or paraphrase them to break the inference chain and anonymize identifying words.
"""


# -- AA: Staab et al.'s adversarial anonymization (2026-10-06) -------------------------------------
# "Large Language Models are Advanced Anonymizers" (ICLR 2025), eth-sri/llm-anonymization (MIT):
# src/anonymized/anonymizers/llm_anonymizers.py LLMFullAnonymizer, prompt_level 3 (the default, "CoT"), and
# src/prompts/prompt.py Prompt.template. Copied verbatim ("text in tact" included).
AA_SYSTEM_PROMPT = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."
AA_HEADER = "Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact."
AA_FOOTER = "In a first step briefly explain the changes you will make. In a new line write a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information."
AA_TEMPLATE = "{header}\n{shots}\n{intermediate}\n\n{footer}\n\n{answer}"


def aa_user_prompt(comments: str, inference_string: str) -> str:
    """LLMFullAnonymizer._create_anon_prompt's user turn (no shots, no answer)."""
    intermediate = f"\n\n {comments}\n\nInferences:\n\n{inference_string}"
    return AA_TEMPLATE.format(header=AA_HEADER, shots="", intermediate=intermediate, footer=AA_FOOTER, answer="")


# -- the evaluation attack ----------------------------------------------------------------------
# TRACE-RPS's attack for its Table 1 (src/reddit/reddit.py create_prompts, the code's only
# implemented path): one prompt per document asking every labelled attribute at once (Staab et
# al.'s "staab_multi"), no Certainty line, rendered with the model_template string of
# configs/reddit/inference_*/llama3_8b.yaml. Byte-identical to the reference (the result tables'
# attack_eval/prompts/staab_*.txt until 2026-10-04).
STAAB_SYSTEM_PROMPT = 'You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible.'
STAAB_USER_TEMPLATE = 'Let us play a guessing game. The following are public comments made on Reddit by one author. Your task is to guess the authors {pii}?\n\n\nComments:\n\n{comments}\n\n{footer}'
STAAB_FOOTER_INTRO = 'First reason step-by-step for each value going over all information in the text as well as language used. Then give your top 3 guesses for each value based on your reasoning. Follow exactly the format specified below:\n\n'
STAAB_FOOTER_ITEM = 'Type: {pii_type}\nInference: Your detailed inference\nGuess: Your top 3 guesses in a list separated by ; in a single line, without any line breaks or numbered lists. {pii_options}\n'

#: llama3_8b.yaml's model_template; tokenized WITH special tokens, which doubles <|begin_of_text|>
#: exactly as the reference's HFModel does
AUTHOR_LLAMA3_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

#: Synthetic records' fixed timestamp 1400463449 -> "2014-05-19"; create_prompts keeps this date
#: prefix on Synthetic comments only (SynthPAI comments are plain)
SYNTHETIC_DATE = "2014-05-19"


def attack_comments(dataset: str, text: str) -> str:
    """create_prompts' comment formatting applied to a "\\n"-joined text: Synthetic lines get the
    "2014-05-19: " prefix. For the original and its masked copies (line by line, after masking)
    and TRACE's rewrite; DP-Fusion paraphrases are free text and are attacked as they are."""
    if dataset != "synthetic":
        return text
    return "\n".join(f"{SYNTHETIC_DATE}: {line.strip()}" for line in text.split("\n"))


def staab_multi_messages(text: str, attributes) -> tuple:
    """(system, user) of the evaluation attack for `text`, asking `attributes` in STAAB_ORDER."""
    from fusit.trace.attributes import ATTRIBUTE_OPTIONS, AUTHOR_KEY, STAAB_ORDER, STAAB_TYPE_STR
    attrs = [a for a in STAAB_ORDER if a in set(attributes)]
    footer = STAAB_FOOTER_INTRO + "".join(
        STAAB_FOOTER_ITEM.format(pii_type=AUTHOR_KEY[a], pii_options=ATTRIBUTE_OPTIONS.get(a, "")) for a in attrs)
    user = STAAB_USER_TEMPLATE.format(pii=", ".join(STAAB_TYPE_STR[a] for a in attrs), comments=text, footer=footer)
    return STAAB_SYSTEM_PROMPT, user.rstrip()


def render_author(system: str, user: str) -> str:
    """AUTHOR_LLAMA3_TEMPLATE filled as HFModel.predict_multi does (user text rstrip()ed);
    tokenize it WITH special tokens."""
    return AUTHOR_LLAMA3_TEMPLATE.format(system=system.strip(), prompt=user.rstrip())


#: TRACE-RPS configs/reddit/inference_*/deepseek.yaml's model_template for DeepSeek-R1-Distill (2026-10-07): user turn
#: only (llmprivacy's HFModel has no system slot, so the system prompt is dropped), the answer opened with "<think>\n";
#: tokenized WITH special tokens, which doubles the BOS as the reference does
DEEPSEEK_R1_TEMPLATE = "<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜><think>\n"


def render_deepseek_r1(user: str) -> str:
    """DEEPSEEK_R1_TEMPLATE filled as HFModel.predict_multi does (prompt rstrip()ed)."""
    return DEEPSEEK_R1_TEMPLATE.format(prompt=user.rstrip())


def render_chat(tokenizer, system: str, user: str, fold_system: bool = False) -> str:
    """The same attack in the attacker's own chat template, for attackers other than Llama-3 Instruct
    (2026-10-07). `fold_system` puts the system prompt at the head of the user turn, for models that
    ask for none (DeepSeek-R1). Tokenize it WITHOUT special tokens: the template writes them."""
    if fold_system:
        messages = [{"role": "user", "content": f"{system.strip()}\n\n{user.rstrip()}"}]
    else:
        messages = [{"role": "system", "content": system.strip()}, {"role": "user", "content": user.rstrip()}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

