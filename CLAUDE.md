# PPIFlow – Codebase Notes for Claude

## Project Overview
Flow-matching framework for de novo protein binder / antibody / nanobody design.
Paper: https://doi.org/10.64898/2026.01.19.700484

## Entry Points

| Script | Task |
|---|---|
| `sample_binder.py` | Binder design against a target chain |
| `sample_antibody_nanobody.py` | Antibody (heavy+light) or nanobody (heavy only) CDR design |
| `sample_monomer.py` | Unconditional monomer generation OR motif scaffolding |
| `sample_antibody_nanobody_partial.py` | Partial-flow refinement for antibody/nanobody |
| `sample_binder_partial.py` | Partial-flow refinement for binder |
| **`pipeline.py`** | **Unified wrapper – run any task from a single YAML file** |

## pipeline.py
- CLI: `python pipeline.py --config <yaml> --output <dir> --num_samples <n> [--resume]`
- Reads `task:` field, builds an `argparse.Namespace`, imports the matching module, calls `module.run_pipeline(args)`.
- After structure generation, automatically runs inverse folding (ProteinMPNN) if `mpnn_weights` is set in the YAML.
- `--resume` skips already-completed steps using `pipeline_state.json` in the output dir.
- Supported task values: `binder`, `antibody`, `nanobody`, `monomer`, `motif_scaffolding`, `partial_flow_antibody`, `partial_flow_nanobody`, `partial_flow_binder`.
- All `_build_*_args` builders, MPNN helpers, and `PipelineState` live in `helper_functions.py`.

## helper_functions.py (added by Claude)
All non-trivial helpers imported by `pipeline.py`:

| Symbol | Purpose |
|---|---|
| `PipelineState` | Reads/writes `pipeline_state.json`; `is_done(step)` / `mark_done(step)` |
| `_require(cfg, *keys)` | Raises clear error if YAML keys are missing |
| `_build_binder_args` | Namespace builder for binder task |
| `_build_antibody_nanobody_args` | Namespace builder for antibody/nanobody |
| `_build_monomer_args` | Namespace builder for monomer/motif_scaffolding |
| `_build_partial_antibody_nanobody_args` | Namespace builder for partial flow antibody/nanobody |
| `_build_partial_binder_args` | Namespace builder for partial flow binder |
| `create_mpnn_fixed_positions_csv(output_dir)` | Scans PDBs by B-factor, writes `mpnn_fixed_positions.csv` |
| `_detect_designed_chains(output_dir)` | Returns chain IDs that have both B-factor 4.0 and 2.0 |
| `run_protein_mpnn(output_dir, csv_path, chain_list, cfg)` | Calls `protein_mpnn_run.main()` directly; output to `<output_dir>/mpnn_output/` |
| `mpnn_fasta_to_csv(input_dirs, output_csv, suffix, top_n=5)` | Reads FASTA files from MPNN seqs dir; deduplicates; keeps top 5 lowest-score sequences per design; writes `seqsfinal_result.csv` with columns `link_name, seq, seq_idx, score` |
| `graft_sequences_to_pdbs(output_dir, csv_path, designed_chains)` | For each row in `seqsfinal_result.csv`, replaces residue names on designed chains in the backbone PDB with the MPNN sequence; writes `mpnn_output/<basename>_<seq_idx>.pdb` |
| `pack_sidechains_dir(input_dir, output_dir, checkpoint)` | Runs FAMPNN sidechain packing on every PDB in input_dir; writes full-atom PDBs to output_dir with PSCE confidence in B-factors |

## Pipeline Steps & State File
Six tracked steps written to `<output_dir>/pipeline_state.json`:
1. `binder_gen` / `nanobody_gen` / etc. – structure sampling
2. `fixed_positions_csv` – `mpnn_fixed_positions.csv` creation
3. `protein_mpnn` – ProteinMPNN inverse folding (FASTA output to `mpnn_output/seqs/`)
4. `fasta_to_csv` – top-5 selection → `mpnn_output/seqsfinal_result.csv`
5. `graft_sequences` – sequence-grafted PDBs written to `mpnn_output/<name>_<seq_idx>.pdb`
6. `fampnn` – FAMPNN sidechain packing → `fampnn_designs/<name>_<seq_idx>.pdb`

## FAMPNN Sidechain Packing
- Weights at `fampnn/weights/`; use `fampnn_0_3.pt` (FAMPNN 3.0, recommended for sequence design)
- Enable by setting `fampnn_weights: <path>` in the pipeline YAML; omit to skip the step
- Input: grafted PDBs from `mpnn_output/`; output: full-atom PDBs in `fampnn_designs/`
- B-factor column in output = PSCE confidence (Å); lower = better sidechain prediction
- `pack_sidechains_dir` in `helper_functions.py` handles the batch loop; `run_fampnn` in `pipeline.py` manages state

`--resume` loads the existing state file and skips any step whose `status == "done"`.

## Inverse Folding (ProteinMPNN)
- `ProteinMPNN/protein_mpnn_run.py` is called as a Python function (not subprocess).
- Fixed positions CSV is tab-separated: `pdb_name<TAB>motif_index`.
- `motif_index` = space-separated 1-based fixed residue indices, `-` separates chains, trailing `-` for single-chain designs.
- Chain detection: designed chains have B-factor 4.0 (framework) AND 2.0 (CDR); binder case falls back to `binder_chain` from config.
- `ProteinMPNN/helper_scripts/make_fixed_positions_dict.py` uses `\t` to split the CSV (updated from original `,`).

## B-Factor Encoding in Output PDBs
| Value | Meaning |
|---|---|
| 4.0 | Antibody/nanobody framework residues (fixed) |
| 2.0 | CDR residues (designed) |
| 1.0 | Antigen hotspot residues |
| 0.0 | Antigen non-hotspot / binder chain (binder task) |

Note: in binder outputs, `hotspot_mask + target_interface_mask` can sum to 2.0 on the antigen chain — designed-chain detection therefore requires **both** 4.0 AND 2.0 to be present on the same chain.

## YAML Keys for ProteinMPNN (all `pipeline_*.yaml`)
| Key | Required | Default |
|---|---|---|
| `mpnn_weights` | Yes (omit to skip MPNN) | – |
| `model_name` | No | `v_48_020` |
| `num_seqs_per_target` | No | `8` |
| `batch_size` | No | `1` |
| `sampling_temp` | No | `"0.1"` |
| `mpnn_omit_AAs` | No | `"X"` |

## Configs Directory (`configs/`)
| File | Used by |
|---|---|
| `inference_binder.yaml` | Binder design |
| `inference_binder_partial.yaml` | Partial-flow binder |
| `inference_nanobody.yaml` | Antibody AND nanobody (same config) |
| `inference_unconditional.yaml` | Monomer unconditional sampling |
| `inference_scaffolding.yaml` | Monomer motif scaffolding |
| `pipeline_*.yaml` | Templates for `pipeline.py` (include MPNN block) |

## Key Architecture Pattern
Every `sample_*.py` follows the same three-step pattern:
1. **Preprocess** – PDB → `.pkl` feature files + CSV manifest
2. **ConfigManager** – loads base YAML config, deep-merges CLI overrides, saves updated YAML to output dir
3. **Experiment / EvalRunner** – `OmegaConf`-based model init → `.test()` / `.run_sampling()`

## Antibody vs Nanobody Mode
Both use `sample_antibody_nanobody.py`. The distinction is:
- **Antibody**: `--light_chain B` is set → processes both heavy and light CDRs
- **Nanobody**: `--light_chain` is omitted / `None` → heavy CDRs only

## CDR Length Format
`"CDRH1,min-max,CDRH2,min-max,CDRH3,min-max"` (comma-separated name/range pairs).
For antibody: also includes CDRL1/L2/L3.
A random length is sampled per run within `[min, max]`.

## Hotspot Format
`"<ChainID><ResNum>,<ChainID><ResNum>,..."` e.g. `"C11,C14,C101"`.
Chain prefix on every residue is required and must match `antigen_chain`.

## Partial Flow `start_t`
- Range: 0.0–1.0
- Higher → preserves more of the input structure (refinement)
- Lower → more stochastic redesign

## Checkpoints
Download from Google Drive (see README). Four checkpoints:
`binder.ckpt`, `antibody.ckpt`, `nanobody.ckpt`, `monomer.ckpt`

## Demo
`demo_vhh.ipynb` – end-to-end VHH pipeline (backbone gen → sequence design → affinity eval).

---

## Residue Interaction Guidance (`models/interaction_guidance.py`)

Added by Claude. Steers the flow-matching generator at inference time so a
specific binder residue type (e.g. ARG) is geometrically favoured to sit
opposite a chosen hotspot residue type (e.g. ASP). **No retraining required.**

### How it works

PPIFlow generates backbone frames only (Cα positions + rotation matrices).
Sequence is assigned afterwards by ProteinMPNN. The guidance nudges the
backbone into conformations where ProteinMPNN is more likely to place the
desired residue, by exploiting the fact that ProteinMPNN is sensitive to
local backbone geometry when choosing which amino acid to place.

At each reverse-diffusion step the model emits a "clean" predicted structure
(`pred_trans_1`, `pred_rotmats_1`). Guidance adds a gradient step on top of
that prediction before the Euler update is taken:

```
score  = geometric_interaction_score(pred_trans_1, ...)
grad   = ∂score / ∂pred_trans_1          # autograd; zeroed on target residues
pred_trans_1 += guidance_scale * (grad / ||grad||)
```

This is the same mechanism as **classifier guidance** in DDPM/flow matching —
the model's trajectory is steered uphill on the reward landscape at every step.

### Score / reward function

For each hotspot residue **h** on the target and each binder residue **b**
within `ca_cutoff` Å (Cα–Cα):

```
pair_score(b, h) = dist_term(b, h)  +  orient_weight × orient_term(b, h)

dist_term   = exp( -0.5 × ((d_CbCb - d_opt) / sigma_d)² )
orient_term = ReLU( cos( CA_b→Cb_b ,  Cb_b→Cb_h ) )
```

- **`d_CbCb`** — Cβ–Cβ distance between binder residue and hotspot.
- **`d_opt`** — optimal Cβ–Cβ distance for the (desired_binder_aa, hotspot_aa)
  pair, from a lookup table of salt bridges, cation-π, and H-bond pairs.
- **`orient_term`** — cosine between the binder's CA→Cβ direction and the
  Cβ_binder→Cβ_hotspot approach direction.  Positive only when the sidechain
  points *toward* the hotspot.
- Per hotspot, the **max** over nearby binder residues is taken (reward one
  good contact, not an average over all).

Cβ positions are estimated from backbone frames:
`Cb ≈ R @ [0.5312, -0.7728, -1.2024] + CA`  (ideal L-amino-acid geometry).
GLY (no Cβ) uses CA instead.

### Optimal Cβ–Cβ distance table

| Interaction | Pair examples | d_opt (Å) |
|---|---|---|
| Salt bridge | ARG–ASP, ARG–GLU, LYS–ASP, LYS–GLU | 6.5 |
| Cation–π    | ARG–PHE, ARG–TRP, ARG–TYR, LYS–PHE, LYS–TYR | 5.5 |
| H-bond      | HIS–ASP, HIS–GLU, ASN–ASP, GLN–GLU | 5.0–5.5 |
| H-bond      | SER–ASP, THR–ASP, SER–GLU, THR–GLU | 4.5 |
| Generic     | any other pair | 6.0 |

### Key symbols

| Symbol | Location | Purpose |
|---|---|---|
| `InteractionGuidance` | `models/interaction_guidance.py` | Main class; call `.apply()` in sampling loop |
| `geometric_interaction_score` | same | Differentiable reward; usable stand-alone for post-hoc ranking or REINFORCE |
| `compute_cb_positions` | same | Estimates Cβ from backbone rigid frames |
| `build_guidance_from_args` | same | Builds guidance object from argparse namespace |

### CLI usage (`scripts/sample_binder.py`)

```bash
python scripts/sample_binder.py \
  --input_pdb target.pdb --target_chain A \
  --specified_hotspots A11,A14 \
  --guidance_binder_aa ARG \       # residue to place on binder
  --guidance_hotspot_aa ASP \      # hotspot type (omit = auto from structure)
  --guidance_scale 2.0 \           # strength; tune 0.5–5.0
  --guidance_start_t 0.5 \         # apply only in 2nd half of trajectory
  --guidance_sigma_d 1.5 \         # Gaussian width (Å)
  --guidance_ca_cutoff 12.0 \      # proximity cutoff (Å)
  --model_weights binder.ckpt --output_dir out/
```

### Tuning guide

| Parameter | Effect of increasing | Recommended start |
|---|---|---|
| `guidance_scale` | Stronger steering; may distort fold at very high values | 1.0 |
| `guidance_start_t` | Lower = guidance applied earlier = stronger global effect | 0.5 |
| `guidance_sigma_d` | Wider distance tolerance = softer reward | 1.5 |
| `guidance_ca_cutoff` | Considers more distant binder residues | 12.0 |

### Integration points

- `data/interpolant_binder.py` — `sample()` accepts `guidance=` kwarg; applies
  `.apply()` after each model call, including the final step.
- `models/flow_module_binder.py` — `test_step` passes `self._guidance` (set
  from outside before calling `.test()`) to `interpolant.sample()`.
- The `score()` method of `InteractionGuidance` returns the scalar reward and
  can serve as a **REINFORCE reward** in a future fine-tuning loop.
