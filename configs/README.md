# configs

Each file holds `fusit.main` arguments (`run:`) and, where needed, what `--decide` builds
(`xpriv:`). A file can `extends:` another and override only what differs. Per-job values stay on
the command line: `--dataset`, `--num-shards` / `--shard-index`, `--output`.

## Running a config

Most configs run in three steps over the same output (all shards of a dataset):

```bash
C=configs/main/implicit.yaml; D=synthpai          # or synthetic
python -m fusit.main --config $C --dataset $D --num-shards N --shard-index i --output out/${D}_$i.jsonl --stop-after verify
python -m fusit.main --config $C --dataset $D --decide out/${D}_*.jsonl     # thresholds over every shard
python -m fusit.main --config $C --dataset $D --num-shards N --shard-index i --output out/${D}_$i.jsonl
python -m fusit.main --config $C --dataset $D --summarize out/${D}_*.jsonl
```

Configs that build on a finished run (everything except `main/implicit.yaml`, `main/tab.yaml`,
`main/implicit_ner_spacy.yaml` and `extra/robust_front_qwen.yaml`) are run on that run's output
(or a copy of it); stages already done are skipped.

## main/ -- the reported experiments

| config | what it runs |
| --- | --- |
| `implicit.yaml` | Synthetic / SynthPAI main run: cue tagging, cue selection, TRACE@1, DP-Fusion (ours), evaluation. K of V_att is per dataset (10 / 30). |
| `implicit_no_defense.yaml` | the original text attacked, judged and scored for PPL |
| `implicit_aa.yaml` | AA@1 |
| `rps.yaml` | RPS on the original, TRACE@1 and every paraphrase (official 2 x 10,000) |
| `implicit_verifier_levels.yaml` | cue-selection thresholds at the 90 / 85 / 80th percentile |
| `implicit_ab_sweep.yaml` | alpha*beta 0.001 ... 10 and inf for NER-only, all cues and ours |
| `implicit_ner_spacy.yaml` | NER (spaCy) mask baseline, in its own output |
| `tab.yaml` | TAB-ECHR (explicit PII): gold NER, every attribute, DP-Fusion's official attack |
| `tab_aa.yaml`, `tab_verifier_levels.yaml` | AA@1 and the threshold sweep on TAB-ECHR |

## extra/ -- analyses and robustness

| config | what it runs |
| --- | --- |
| `implicit_ablation.yaml` | X_priv component ablation and the coverage-matched random controls (built by `--decide` with this config) |
| `implicit_ablation_noncue.yaml` | random words outside every cue at the same coverage |
| `implicit_control_vcotrand.yaml`, `tab_control_vcotrand.yaml` | random V_cot words at the coverage of each threshold level |
| `implicit_ab_fine.yaml` | alpha*beta 0.01 ... 0.10 for ours |
| `implicit_katt_sweep.yaml` | V_att top-K sensitivity: add `--k-att K`; add `--judge` in a second pass for utility |
| `implicit_vatt_verified.yaml`, `tab_vatt_verified.yaml` | V_att words through cue selection too (`--verify-att`) |
| `sample_katt30.yaml` | K = 30 on a 50-profile SynthPAI sample |
| `robust_front_qwen.yaml` | Qwen2.5-7B-Instruct as the inference model (defence side of combinations 1 and 3) |
| `robust_combo_qwen_attack.yaml` | attack side of combinations 1 and 3, on a copy of the `robust_front_qwen` run |
| `robust_combo1_llama2_7b*.yaml` | combination 1: attacker Llama-2-7B-Chat (`_kvofficial`: RPS at 2 x 10,000) |
| `robust_combo2_deepseek_qwen7b.yaml` | combination 2: attacker DeepSeek-R1-Distill-Qwen-7B on the main run's texts |
| `robust_combo2_deepseek_qwen7b_rps.yaml` | its RPS rows, in three passes: `--skip-attack` (RPS search), as is (attack), `--judge --dtype float16` (judge) |
| `robust_combo3_llama2_13b*.yaml` | combination 3: attacker Llama-2-13B-Chat over two GPUs (`--device auto`) |
| `robust_attack_*.yaml`, `robust_rps_*.yaml` | the main run's texts attacked, and RPS re-optimized, against other attackers |

Notes:
- alpha*beta values are written as floats on purpose: config values skip argparse's `type=float`,
  and an int `1` would key a paraphrase `@1` while the perplexity stage keys it `@1.0`.
- RPS search and the attack can run in separate processes (`--skip-attack`, then the same config
  without it) when the two models do not fit on one GPU together.
