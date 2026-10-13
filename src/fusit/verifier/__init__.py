"""
Cue selection (the verifier): which words of the V_cot cues actually move the inference model's
prediction of an attribute, so that X_priv keeps those and drops the rest.

It works on the records of the cue tagger (fusit.trace.CueTagger.tag) and never generates cues
itself:

    candidates  the value set p(value | text) is taken over (all options / the top-3 guesses)
    scoring     p(value | text) by the candidates' log-likelihood after a direct-answer prompt,
                and the divergences built on it (JSD, entropy)
    units       V_cot quotes -> content-word units; the null (random) words
    score       per (item, attribute): the per-token log-probs of D, the empty text,
                D minus each unit / null word, and each unit / null word alone
    decide      S_cond = JSD(p(D) || p(D minus u)), S_alone = JSD(p(empty) || p(u)); null
                thresholds per (dataset, attribute) at a percentile; OR rule
    xpriv       X_priv per condition from NER, V_att and the (selected) V_cot words
"""
