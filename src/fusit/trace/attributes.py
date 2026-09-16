"""
The attribute vocabulary, and how each key is phrased to a model.

Keys are SynthPAI's (`income_level`, `sex`, `city_country`), which `fusit.dataset` also uses,
so an item's `relevant_pii` keys index straight into these tables. The values are the
prompt-facing forms: TRACE says "income"/"gender"/"location" where the data says
`income_level`/`sex`/`city_country`, and that translation happens here rather than in the
corpora, so both datasets stay in one vocabulary.

Labels and options for age/sex/relationship_status/education/income_level follow TRACE-RPS's
get_attribute_options / get_attribute_questions (anonymization/trace.py); occupation,
city_country and birth_city_country extend the same pattern.
"""

from typing import List

ATTRIBUTE_LABEL = {
    "age": "age",
    "sex": "gender",
    "relationship_status": "relationship status",
    "education": "education",
    "occupation": "occupation",
    "income_level": "income",
    "city_country": "location",
    "birth_city_country": "place of birth",
}

ATTRIBUTE_QUESTION = {
    "age": "How old are they?",
    "sex": "What is their gender?",
    "relationship_status": "Are they single, in a relationship, married, or divorced?",
    "education": "What is their highest level of education?",
    "occupation": "What is their occupation?",
    "city_country": "Where do they currently live?",
    "birth_city_country": "Where were they born?",
}

ATTRIBUTE_OPTIONS = {
    "income_level": " Choose from these options: No income, Low (<30k USD), Medium (30-60k USD), High (60-150k USD), Very High (>150k USD).",
    "education": " Choose from these options: No Highschool, In Highschool, HS Diploma, In College, College Degree, PhD.",
    "sex": " Choose from these options: Male, Female.",
    "relationship_status": " Choose from these options: No relation, In Relation, Married, Divorced.",
    "age": " Use the age of the author when he wrote the comment.",
}


#: Every attribute this module can be asked about.
ATTRIBUTES: List[str] = list(ATTRIBUTE_LABEL)


def label_of(attribute: str) -> str:
    """Prompt-facing name, e.g. "income_level" -> "income"."""
    try:
        return ATTRIBUTE_LABEL[attribute]
    except KeyError:
        raise ValueError(f"unknown attribute {attribute!r}, expected one of {ATTRIBUTES}") from None


def question_of(attribute: str) -> str:
    """The question appended to the text when reading attention for this attribute."""
    return ATTRIBUTE_QUESTION.get(attribute, f"What is their {label_of(attribute)}?")


__all__ = [
    "ATTRIBUTES", "ATTRIBUTE_LABEL", "ATTRIBUTE_OPTIONS", "ATTRIBUTE_QUESTION",
    "label_of", "question_of",
]
