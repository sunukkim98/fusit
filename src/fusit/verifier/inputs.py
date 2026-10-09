"""
What the verifier reads: the tag stage's record (fusit.trace.CueTagger.tag) in the verifier's
per-attribute form. Nothing is generated here -- the guesses, chains and located quotes are
TRACE's, as the tagger produced them.

    verifier_cues(tagged) -> {attribute: {"guess": {"guesses"}, "c": [{c_id, text, spans, quotes}],
                                          "att": [{"word", "span"}]}}

A candidate c is a located quote string (case-insensitively deduplicated); its spans are the union
of where its quotes were located (every occurrence, as the tagger counted them). Quotes the tagger
could not locate carry no c.
"""

from typing import Dict

from fusit.trace.spans import merge_spans


def verifier_cues(tagged: Dict) -> Dict[str, Dict]:
    out = {}
    for attr, rec in tagged["attributes"].items():
        cot = rec.get("cot", {})
        by_key = {}
        for q in cot.get("quotes", []):
            if not q.get("located") or not q["spans"]:
                continue
            entry = by_key.setdefault(q["located"].lower(), {"text": q["located"], "spans": [], "quotes": []})
            entry["spans"] += q["spans"]
            entry["quotes"].append(q["quote"])
        att = rec.get("att", {"words": [], "spans": []})
        out[attr] = {
            "guess": {"guesses": list(cot.get("guesses", [])), "inference": cot.get("inference", ""),
                      "certainty": cot.get("certainty")},
            "c": [{"c_id": i, "text": e["text"], "spans": merge_spans(e["spans"]), "quotes": e["quotes"]}
                  for i, e in enumerate(by_key.values())],
            "att": [{"word": w, "span": list(sp)} for w, sp in zip(att["words"], att["spans"])],
        }
    return out


__all__ = ["verifier_cues"]
