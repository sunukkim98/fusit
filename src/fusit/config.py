"""
Experiment configuration files (configs/*.yaml, 2026-10-05).

A config names what an experiment runs; the package only says what can be run. Two sections:

    run:     defaults for fusit.main's options, by their long name ("ner-backend" or "ner_backend").
             Anything given on the command line wins, so per-job values (dataset, shard, output)
             stay on the command line and a config never has to change between shards.
    xpriv:   what --decide builds besides the core X_priv conditions (fusit.verifier.xpriv):
                 verified_levels   the decision levels turned into ner_att_vcot<level> (default: all decided)
                 verified_att      true: also ner_vatt_vcot<level>, V_att through the verifier too
                                   (needs the units of --verify-att; 2026-10-06)
                 controls          experiment-only conditions, each {kind, levels, seeds}:
                                   words_all     ner_att_vcotwords (every V_cot word unit, no verifier)
                                   random_doc    random_cov<level>_s<seed>: random content words outside
                                                 NER u V_att, coverage matched to ner_att_vcot<level>
                                   random_vcot   ner_att_vcotrand<level>_s<seed>: random V_cot word units,
                                                 coverage matched to ner_att_vcot<level>
                                   random_noncue random_noncue<level>_s<seed>: as random_doc with V_cot also
                                                 out of the pool -- words that are no cue (2026-10-06)

`extends: other.yaml` (relative to the file) loads that config first and deep-merges this one over it,
so an experiment config only states what differs from the run it builds on.

A `run` value may be a mapping by dataset, `k_att: {synthetic: 10, synthpai: 30, tab: 10}`: the entry
of the run's --dataset (command line, else the config's) is used (2026-10-09).
"""

from pathlib import Path
from typing import Dict, Optional

SECTIONS = ("run", "xpriv")


def _merge(base: Dict, over: Dict) -> Dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path, _seen: Optional[set] = None) -> Dict:
    """{section: values} of `path`, with its `extends` chain resolved."""
    import yaml

    path = Path(path).resolve()
    seen = _seen or set()
    if path in seen:
        raise ValueError(f"config extends itself: {path}")
    seen.add(path)
    raw = yaml.safe_load(path.read_text()) or {}
    unknown = set(raw) - set(SECTIONS) - {"extends", "description"}
    if unknown:
        raise ValueError(f"{path}: unknown sections {sorted(unknown)}; expected {SECTIONS}")
    base = load_config(path.parent / raw["extends"], seen) if raw.get("extends") else {}
    return _merge(base, {k: raw.get(k) or {} for k in SECTIONS if k in raw})


def run_defaults(cfg: Dict, parser) -> Dict:
    """The config's `run` section as argparse defaults: option names checked against `parser`."""
    dests = {a.dest for a in parser._actions}
    out = {}
    for key, value in (cfg.get("run") or {}).items():
        dest = key.replace("-", "_")
        if dest not in dests:
            raise ValueError(f"config run.{key}: fusit.main has no option --{dest.replace('_', '-')}")
        out[dest] = value
    return out


def per_dataset(defaults: Dict, dataset: str) -> Dict:
    """The {dataset: value} entries of `defaults` (from `run_defaults`), resolved for `dataset`."""
    out = {}
    for dest, value in defaults.items():
        if isinstance(value, dict):
            if dataset not in value:
                raise ValueError(f"config run.{dest}: no value for dataset {dataset!r} (has {sorted(value)})")
            out[dest] = value[dataset]
    return out


__all__ = ["load_config", "per_dataset", "run_defaults"]
