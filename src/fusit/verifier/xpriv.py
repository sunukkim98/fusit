"""
X_priv per condition for one item, from NER, the tagger's records and the unit decisions.
Spans are char offsets into D, merged per source.

Conditions:
    ner_only, ner_cot, ner_att, ner_both           NER / V_att / V_cot (quote level, all occurrences)
    ner_att_vcot95, ner_att_vcot90                 NER u V_att u verified V_cot words (all occurrences)
    random_cov95_s{seed}                           NER u V_att u random content words, per-item char
                                                   coverage matched to ner_att_vcot95
    ner_att_vcotwords                              NER u V_att u EVERY V_cot word unit -- word
                                                   splitting without the verifier
    ner_att_vcotrand{95,90,85,80}_s{seed}          NER u V_att u V_cot word units drawn at random (seed
                                                   0-2) until they add as many chars as the verified
                                                   units of that level do: the coverage-matched control
                                                   of the verifier's choice (2026-10-05)
    random_noncue{level}_s{seed}                   as random_cov, but V_cot (quotes and every occurrence of a
                                                   V_cot word) is out of the pool too: random words that are
                                                   no cue at all (2026-10-06)
    ner_vatt_vcot{level}                           NER u verified V_att words u verified V_cot words
                                                   (2026-10-06): V_att through the verifier too; needs
                                                   units scored with --verify-att and `verified_att: true`
    ner_vatt_cot{level}                            NER u verified V_att words u every V_cot quote (as
                                                   ner_both): the V_att-only arm of the 2 x 2 (2026-10-06)

The first two groups are core; the rest are experiment-only and are made only when a config's
`xpriv` section asks for them (fusit.config). Without a spec, build_item makes LEGACY_SPEC's set.
Units carry their source ("cot" or "att"; units scored before 2026-10-06 are all V_cot). Every condition
but ner_vatt_vcot reads the V_cot units only, so adding V_att units changes none of them.
"""

import random
from typing import Dict, List, Optional

from fusit.trace.spans import WORD_RE, merge_spans
from fusit.utils import seed_for
from fusit.verifier.units import content_words, core

BASELINES = {"ner_only": ("ner",), "ner_cot": ("ner", "cot"), "ner_att": ("ner", "att"),
             "ner_both": ("ner", "att", "cot")}
RANDOM_SEEDS = (0, 1, 2)


def spans_of_att(rec: dict) -> List[List[int]]:
    """V_att spans of one attribute's cue record: the selected words."""
    return [w["span"] for w in rec.get("att", []) if w["span"][1] > w["span"][0]]


def spans_of_cot(rec: dict) -> List[List[int]]:
    """V_cot spans of one attribute's cue record: every occurrence of every matched quote."""
    return [s for c in rec.get("c", []) for s in c["spans"]]


def coverage_stats(text: str, spans) -> dict:
    """Char fraction and word fraction (a whitespace word counts when any of its chars is
    covered) of the merged spans."""
    merged = merge_spans([list(s) for s in spans])
    covered = set()
    for s, e in merged:
        covered.update(range(s, e))
    words = [(m.start(), m.end()) for m in WORD_RE.finditer(text)]
    hit = sum(1 for s, e in words if any(i in covered for i in range(s, e)))
    return {"char": len(covered) / len(text) if text else 0.0,
            "word": hit / len(words) if words else 0.0}


def source_spans(cues: Dict[str, dict], ner_spans: List[List[int]],
                 accepted_cot: Optional[List[List[int]]] = None) -> Dict[str, List[List[int]]]:
    """{"ner", "att", "cot"} spans of one item. "cot" is every V_cot quote occurrence, or
    `accepted_cot` (the verified word spans) when given."""
    att, cot = [], []
    for a in cues.values():
        att += spans_of_att(a)
        cot += spans_of_cot(a)
    return {"ner": merge_spans(ner_spans), "att": merge_spans(att),
            "cot": merge_spans(cot if accepted_cot is None else accepted_cot)}


def union(src: Dict[str, List[List[int]]], sources) -> List[List[int]]:
    return merge_spans([s for k in sources for s in src[k]])


def chars(spans) -> set:
    out = set()
    for s, e in merge_spans([list(x) for x in spans]):
        out.update(range(s, e))
    return out


def random_extra(text: str, base: List[List[int]], want: int, seed: int) -> List[List[int]]:
    """Random content words outside `base`, added in random order until they cover `want` chars
    (the last one cut to land exactly). Stops short when the pool runs out."""
    taken = chars(base)
    pool = [[s, e] for s, e, _ in content_words(text) if not any(i in taken for i in range(s, e))]
    random.Random(seed).shuffle(pool)
    out, got = [], 0
    for s, e in pool:
        if got >= want:
            break
        e = min(e, s + (want - got))
        out.append([s, e])
        got += e - s
    return out


def random_units(units: List[dict], base: List[List[int]], want: int, seed: int) -> List[List[int]]:
    """V_cot word units (each with all its occurrences, as the verifier decides them) in random
    order, added until the chars they add beyond `base` reach `want`; the last unit is kept only if
    that lands closer to `want`. Units are never cut. A word decided under several attributes
    counts once."""
    taken = chars(base)
    pool, seen = [], set()
    for u in units:
        key = tuple(map(tuple, u["spans"]))
        if key not in seen:
            seen.add(key)
            pool.append(u["spans"])
    random.Random(seed).shuffle(pool)
    out, got = [], set()
    for spans in pool:
        if len(got) >= want:
            break
        new = chars(spans) - taken - got
        if not new:
            continue
        if len(got) + len(new) > want and (len(got) + len(new) - want) > (want - len(got)):
            break                                  # overshooting would land further from want
        out += [list(x) for x in spans]
        got |= new
    return merge_spans(out)


#: What build_item made before configs existed (2026-10-05): every decided level, the all-words
#: condition and the document-wide random control at @95. Used when no `xpriv` spec is given.
LEGACY_SPEC = {"verified_levels": None,
               "controls": [{"kind": "words_all"}, {"kind": "random_doc", "levels": [95], "seeds": list(RANDOM_SEEDS)}]}
LEVELS = (95, 90, 85, 80)
CONTROL_KINDS = ("words_all", "random_doc", "random_vcot", "random_noncue")


def build_item(dataset: str, item_id: str, text: str, cues: Dict[str, dict], ner_spans: List[List[int]],
               units: List[dict], spec: Optional[dict] = None) -> Dict[str, dict]:
    """{condition: {"spans": {source: spans}, "x_priv": merged, "coverage"}} for one item.
    `units` are its decided word units ({attribute, text, spans, decision@<level>, ...}).

    Core conditions, always: ner_only / ner_cot / ner_att / ner_both, and ner_att_vcot<level> for
    `spec["verified_levels"]` (default: every decided level). Experiment-only conditions come from
    `spec["controls"]` (fusit.config, configs/*.yaml); without a spec, LEGACY_SPEC."""
    spec = LEGACY_SPEC if spec is None else spec
    all_units = units
    units = [u for u in all_units if u.get("source", "cot") == "cot"]      # V_cot units: every other condition
    att_units = [u for u in all_units if u.get("source") == "att"]
    src = source_spans(cues, ner_spans)
    att_words = {(a, core(0, w["word"])[2].lower()) for a, rec in cues.items() for w in rec.get("att", [])}
    words_all = merge_spans([s for u in units for s in u["spans"]])
    out = {name: {"ner": src["ner"], "att": src["att"] if "att" in s else [], "cot": src["cot"] if "cot" in s else []}
           for name, s in BASELINES.items()}
    # word-level candidate set of this item: every content word of every V_cot quote
    vocab = {(a, w.lower()) for a, att in cues.items()
             for c in att["c"] for sp in c["spans"] for _, _, w in content_words(text, *sp)}
    decided = [lv for lv in LEVELS if all(f"decision@{lv}" in u for u in all_units)]
    levels = decided if spec.get("verified_levels") is None else list(spec["verified_levels"])
    missing = set(levels) - set(decided)
    if missing:
        raise ValueError(f"{item_id}: levels {sorted(missing)} were not decided (fusit.verifier.decide)")
    for lv in levels:
        key = f"decision@{lv}"
        acc = [u for u in units if u[key]]
        assert all((u["attribute"], u["text"].lower()) in vocab for u in acc), f"{item_id}: verified word outside V_cot"
        cot = merge_spans([s for u in acc for s in u["spans"]])
        assert chars(cot) <= chars(words_all), f"{item_id}: verified spans not within V_cot word spans"
        out[f"ner_att_vcot{lv}"] = {"ner": src["ner"], "att": src["att"], "cot": cot}
        if spec.get("verified_att"):
            # V_att words that are stopwords or already V_cot units have no V_att unit of their own: the former
            # drop out (the unit rule), the latter follow their V_cot unit's decision
            vatt = merge_spans([s for u in att_units if u[key] for s in u["spans"]])
            out[f"ner_vatt_vcot{lv}"] = {"ner": src["ner"], "att": vatt, "cot": cot}   # (those are in `cot`)
            # V_att verified, V_cot whole (quote level, as ner_both): the overlap words' V_cot decision is added
            # to the V_att side, since the whole-V_cot side does not carry their occurrences outside quotes
            over = [u for u in acc if (u["attribute"], u["text"].lower()) in att_words]
            out[f"ner_vatt_cot{lv}"] = {"ner": src["ner"], "att": merge_spans(vatt + [s for u in over for s in u["spans"]]),
                                        "cot": src["cot"]}

    base = union(src, ("ner", "att"))

    def added(lv):
        """chars ner_att_vcot<lv> adds beyond NER u V_att: what a coverage-matched control adds"""
        acc = [u for u in units if u[f"decision@{lv}"]]
        return len(chars(base + merge_spans([s for u in acc for s in u["spans"]]))) - len(chars(base))

    for c in spec.get("controls") or []:
        kind = c["kind"]
        if kind not in CONTROL_KINDS:
            raise ValueError(f"unknown xpriv control {kind!r}, expected one of {CONTROL_KINDS}")
        if kind == "words_all":
            out["ner_att_vcotwords"] = {"ner": src["ner"], "att": src["att"], "cot": words_all}
            continue
        for lv in c.get("levels", [95]):
            if lv not in decided:
                raise ValueError(f"control {kind}: level {lv} was not decided")
            want = added(lv)
            for seed in c.get("seeds", list(RANDOM_SEEDS)):
                if kind == "random_doc":
                    # @95 keeps the seed the results so far were drawn with
                    rs = seed_for(dataset, item_id, seed) if lv == 95 else seed_for(dataset, item_id, "random_doc", lv, seed)
                    out[f"random_cov{lv}_s{seed}"] = {"ner": src["ner"], "att": src["att"],
                                                      "random": random_extra(text, base, want, rs)}
                elif kind == "random_noncue":
                    # random_doc without the cues: V_cot (every quote occurrence and every occurrence of a V_cot word)
                    # is left out of the pool as well as NER u V_att
                    noncue = merge_spans(base + src["cot"] + words_all)
                    out[f"random_noncue{lv}_s{seed}"] = {
                        "ner": src["ner"], "att": src["att"],
                        "random": random_extra(text, noncue, want, seed_for(dataset, item_id, "random_noncue", lv, seed))}
                else:
                    out[f"ner_att_vcotrand{lv}_s{seed}"] = {
                        "ner": src["ner"], "att": src["att"],
                        "cot": random_units(units, base, want, seed_for(dataset, item_id, "vcotrand", lv, seed))}
    return {name: {"spans": sp, "x_priv": merge_spans([s for v in sp.values() for s in v]),
                   "coverage": coverage_stats(text, [s for v in sp.values() for s in v])} for name, sp in out.items()}


__all__ = ["BASELINES", "CONTROL_KINDS", "LEGACY_SPEC", "LEVELS", "RANDOM_SEEDS", "build_item", "chars", "random_extra", "random_units", "source_spans", "union"]
