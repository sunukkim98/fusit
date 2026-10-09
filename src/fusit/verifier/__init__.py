"""
The cue verifier: which of TRACE's V_cot cues actually move an attacker's belief about the
attribute, so that X_priv keeps those and drops the rest.

It works on what fusit's cue tagger already produced (fusit.trace.CueTagger.tag records, the
pipeline's "tag" stage) and never generates cues itself:

    candidates  the value set p(value | text) is taken over (closed options / the CoT's top-3)
    scoring     p(value | text) by the candidates' log-likelihood after a direct-answer prompt,
                and the divergences built on it (JSD, entropy)
    units       V_cot quotes -> content-word units; the null words
    score       per (item, attribute): the stored per-token log-probs of D, the empty text,
                D minus each unit / null word, and each unit / null word alone
    decide      S_cond = JSD(p(D) || p(D minus u)), S_alone = JSD(p(empty) || p(u)); null
                thresholds per (dataset, attribute) at the 95th percentile; OR rule
    xpriv       X_priv per condition from NER, V_att and the (verified) V_cot

Moved from verifier/ on 2026-10-04 unchanged; verifier/ re-exports these modules for the
analysis scripts that still import them from there.
"""
