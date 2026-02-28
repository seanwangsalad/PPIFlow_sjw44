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

## Pipeline Steps & State File
Three tracked steps written to `<output_dir>/pipeline_state.json`:
1. `binder_gen` – structure sampling
2. `fixed_positions_csv` – `mpnn_fixed_positions.csv` creation
3. `protein_mpnn` – ProteinMPNN inverse folding

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
