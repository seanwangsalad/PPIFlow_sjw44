# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# PPIFlow – Codebase Notes for Claude

## Commands

```bash
# Environment
conda activate ppiflow

# Run any task via pipeline (recommended)
python pipeline.py --config configs/pipeline_nanobody.yaml --output /path/to/output --num_samples 5

# Resume a previous run (skips completed steps)
python pipeline.py --config configs/pipeline_nanobody.yaml --output /path/to/output --num_samples 5 --resume

# Run scripts directly (bypasses pipeline.py)
python scripts/sample_binder.py \
    --input_pdb /path/to/target.pdb \
    --target_chain B --binder_chain A \
    --config configs/inference_binder.yaml \
    --model_weights weights/binder.ckpt \
    --output_dir /path/to/output --name myrun --samples_per_target 5
```

No formal test suite. The `test/` directory contains example output runs, not executable tests.

## Scope Notice

**Only initial structure generation and partial flow tasks are supported.** All other modules (ProteinMPNN, FAMPNN, downstream scoring, etc.) are not supported and will be removed in a future cleanup.

Supported tasks: `binder`, `antibody`, `nanobody`, `monomer`, `motif_scaffolding`, `partial_flow_ab`, `partial_flow_binder`.

## Project Overview
Flow-matching framework for de novo protein binder / antibody / nanobody backbone generation.
Paper: https://doi.org/10.64898/2026.01.19.700484

PPIFlow generates backbone PDBs only. Inverse folding, sidechain packing, scoring, and
affinity maturation are handled downstream by **barbarossa** (`../barbarossa`).

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
- Reads `task:` field from YAML, builds an `argparse.Namespace`, imports the matching module, calls `module.run_pipeline(args)`.
- `--resume` scans the output directory for existing `{name}_{int}.pdb` files and continues from where generation stopped.

### Supported task values

| Task | Module |
|---|---|
| `binder` | `scripts.sample_binder` |
| `antibody` | `scripts.sample_antibody_nanobody` |
| `nanobody` | `scripts.sample_antibody_nanobody` |
| `monomer` | `scripts.sample_monomer` |
| `motif_scaffolding` | `scripts.sample_monomer` |
| `partial_flow_ab` | `scripts.sample_antibody_nanobody_partial` |
| `partial_flow_binder` | `scripts.sample_binder_partial` |

`partial_flow_ab` covers both antibody and nanobody refinement — omit `light_chain` in the YAML for nanobody mode.

## helper_functions.py
Minimal shared utilities imported by `pipeline.py`.

| Symbol | Purpose |
|---|---|
| `_require(cfg, *keys)` | Raises clear error if YAML keys are missing |
| `_build_binder_args` | Namespace builder for binder task |
| `_build_antibody_nanobody_args` | Namespace builder for antibody/nanobody |
| `_build_monomer_args` | Namespace builder for monomer/motif_scaffolding |
| `_build_partial_antibody_nanobody_args` | Namespace builder for partial_flow_ab |
| `_build_partial_binder_args` | Namespace builder for partial_flow_binder |

Notes:
- There is no longer a `PipelineState` abstraction or `pipeline_state.json`; resume behavior is implemented only for binder and antibody/nanobody generation via output PDB filename counting.
- `pipeline.py` only calls `validate_inputs(args)` when the target script defines it. In the current tree, `scripts/sample_binder.py` and `scripts/sample_monomer.py` define it; the partial-flow and antibody/nanobody generation scripts do not.

## Output
```
<output_dir>/
└── *.pdb                  # backbone PDBs (B-factor encoded)
```

Only backbone PDBs are generated. No MPNN, FAMPNN, AF3Score, or FastRelax output.

## Chain ID Convention in Output PDBs

Output PDB chain IDs match the `target_chain` / `binder_chain` values specified in the YAML — for all tasks (generation, partial flow) and all molecule types (binder, antibody, nanobody).

**Binder tasks** (`binder`, `partial_flow_binder`): internally chains are remapped to 0=target / 1=binder during inference, then remapped back to the user-specified chain IDs at write time (`models/flow_module_binder.py`, `models/flow_module_binder_partial.py`). The `target_id` and `binder_id` fields are propagated through the batch from the dataset CSV for this purpose.

**Antibody/nanobody tasks**: chain IDs are preserved directly from the input PDB throughout; no remapping needed.

## B-Factor Encoding in Output PDBs

| Value | Meaning |
|---|---|
| 4.0 | Antibody/nanobody framework residues (fixed) |
| 2.0 | CDR residues (designed) |
| 1.0 | Antigen hotspot residues |
| 0.0 | Antigen non-hotspot / binder non-designed residues |

Note: in binder outputs, `hotspot_mask + target_interface_mask` can sum to 2.0 on the antigen chain — designed-chain detection requires **both** 4.0 AND 2.0 to be present on the same chain.

**Partial flow backbone PDBs** (from `sample_antibody_nanobody_partial.py`) use:

| Value | Meaning |
|---|---|
| 1.0 | Binder framework (structurally fixed) |
| 0.0 | Binder CDR (designed) |
| 2.0 | Antigen hotspot |
| 0.0 | Antigen non-hotspot |

## Configs Directory (`configs/`)

| File | Used by |
|---|---|
| `inference_binder.yaml` | Binder design |
| `inference_binder_partial.yaml` | Partial-flow binder |
| `inference_nanobody.yaml` | Antibody AND nanobody (same config) |
| `inference_unconditional.yaml` | Monomer unconditional sampling |
| `inference_scaffolding.yaml` | Monomer motif scaffolding |
| `pipeline_binder.yaml` | Template for binder task |
| `pipeline_antibody.yaml` | Template for antibody task |
| `pipeline_nanobody.yaml` | Template for nanobody task |
| `pipeline_monomer.yaml` | Template for monomer / motif_scaffolding |
| `pipeline_partial_flow_antibody.yaml` | Template for partial_flow_ab task |
| `pipeline_partial_flow_binder.yaml` | Template for partial_flow_binder task |

## YAML Keys (all pipeline_*.yaml)

### Required for all tasks
| Key | Description |
|---|---|
| `task` | One of the supported task values above |
| `model_weights` | Path to `.ckpt` checkpoint |
| `config` | Path to inference YAML (e.g. `configs/inference_nanobody.yaml`) |
| `name` | Run name prefix for output PDB filenames |

### Binder / partial_flow_binder
| Key | Required | Notes |
|---|---|---|
| `input_pdb` | Yes (or `input_csv`) | Target PDB |
| `input_csv` | Yes (or `input_pdb`) | Preprocessed CSV |
| `target_chain` | No | Chain ID of target |
| `binder_chain` | No | Chain ID of designed binder |
| `specified_hotspots` | No | e.g. `"C31,C63,C78"` |
| `samples_min_length` | No | Min binder length |
| `samples_max_length` | No | Max binder length |
| `fixed_positions` | No (partial only) | e.g. `"A97-111"` |
| `start_t` | No (partial only) | 0.0–1.0; higher = more structure preserved |

### Antibody / Nanobody / partial_flow_ab
| Key | Required | Notes |
|---|---|---|
| `antigen_pdb` | Yes (gen only) | Antigen structure |
| `scaffold_pdb` | Yes¹ (gen only) | Complete scaffold PDB (all residues present); use with `scaffold_redesign_residues` + `scaffold_redesign_lengths` |
| `scaffold_redesign_residues` | Yes¹ (gen only) | Inclusive residue ranges to redesign, e.g. `[[26,33],[51,57],[96,110]]` |
| `scaffold_redesign_lengths` | Yes¹ (gen only) | `[min,max]` length per region, e.g. `[[8,8],[8,8],[9,21]]` |
| `scaffold_redesign_chains` | No (gen only) | Scaffold chain ID per region, e.g. `[A,A,A,B,B,B]`. Omit for nanobody — defaults to the scaffold's first chain. Required for antibody when redesign regions span both chains. |
| `framework_pdb` | Yes² (gen only) | Pre-trimmed framework PDB (legacy; CDR loops already removed, gaps present) |
| `complex_pdb` | Yes (partial only) | Full antibody-antigen complex PDB |
¹ Scaffold mode: provide `scaffold_pdb` + `scaffold_redesign_residues` + `scaffold_redesign_lengths` (pipeline trims internally).
² Legacy mode: provide pre-trimmed `framework_pdb` + `cdr_length` string.

| `antigen_chain` | Yes | Chain ID of antigen |
| `heavy_chain` | Yes | Chain ID of heavy chain |
| `light_chain` | No | Omit for nanobody mode |
| `specified_hotspots` | No | e.g. `"C11,C14,C101"` |
| `cdr_length` | No (gen only) | e.g. `"CDRH1,8-8,CDRH3,10-20"` |
| `cdr_position` | No (partial only) | e.g. `"A26-33,A51-58"` |
| `fixed_positions` | Yes (partial only) | Residues that must not move |
| `start_t` | Yes (partial only) | 0.0–1.0 |
| `retry_Limit` | No (partial only) | Default: 10 |

### Monomer / motif_scaffolding
| Key | Required | Notes |
|---|---|---|
| `length_subset` | No | Length range |
| `motif_csv` | No (scaffolding only) | Motif definitions |
| `motif_names` | No (scaffolding only) | Names to use |

## Antibody vs Nanobody Mode
Both use `sample_antibody_nanobody.py`. The distinction:
- **Antibody**: `light_chain: B` is set → processes both heavy and light CDRs
- **Nanobody**: `light_chain` omitted / `null` → heavy CDRs only

Same applies to `partial_flow_ab` — omit `light_chain` for nanobody refinement.

## CDR Length Format
`"CDRH1,min-max,CDRH2,min-max,CDRH3,min-max"` (comma-separated name/range pairs).
For antibody: also includes CDRL1/L2/L3. A random length is sampled per run within `[min, max]`.

## Hotspot Format
`"<ChainID><ResNum>,<ChainID><ResNum>,..."` e.g. `"C11,C14,C101"`.
Chain prefix on every residue is required and must match `antigen_chain`.

## Partial Flow `start_t`
- Range: 0.0–1.0
- Higher → preserves more of the input structure (refinement)
- Lower → more stochastic redesign
- Why `partial_flow_ab` and `partial_flow_binder` are separate: different `Experiment` classes (`inference_antibody_partial` vs `inference_binder_partial`), different preprocessing pipelines, different config structures.

## Scaffold Mode Chain Handling (`sample_antibody_nanobody.py`)

`_build_redesigned_scaffold` and `_generate_framework_reference` both:
- Read the scaffold's **actual chain IDs** from the PDB (not from the YAML)
- Remap to user-specified chain IDs in the output: scaffold chain → `heavy_chain` (nanobody: all chains; antibody: first scaffold chain → heavy, second → light)
- Use `scaffold_redesign_chains` as **scaffold chain IDs** (not output chain IDs) when matching redesign regions

This means `heavy_chain: B` with a scaffold that has chain `A` works correctly — the output PDB uses chain `B` as specified.

The preprocessing CSV (`{name}_input.csv`) is deleted at the start of each non-resume run to prevent stale rows from accumulating across failed runs.

The framework reference PDB (`{name}_framework_ref.pdb`) is written to `<output_dir>/input/` alongside the PKL files and CSV.

`calc_rmsd` in `analysis/antibody_metric.py` returns `inf` (instead of raising) when CA atom counts differ between filtered output and framework reference. This is a known edge case where 1–2 framework residues with near-zero normalized coordinates are excluded from the output PDB by the `atom37_mask` threshold.

## Key Architecture Pattern
Every `sample_*.py` follows the same three-step pattern:
1. **Preprocess** – PDB → `.pkl` feature files + CSV manifest
2. **ConfigManager** – loads base YAML config, deep-merges CLI overrides, saves updated YAML to output dir
3. **Experiment / EvalRunner** – `OmegaConf`-based model init → `.test()` / `.run_sampling()`

## Checkpoints
Download from Google Drive (see README). Four checkpoints:
`binder.ckpt`, `antibody.ckpt`, `nanobody.ckpt`, `monomer.ckpt`

## Demo
`demo_vhh.ipynb` – end-to-end VHH pipeline (backbone gen → sequence design → affinity eval).

## Downstream Processing
After generating backbone PDBs, hand off to **barbarossa** (`../barbarossa`) for sequence design, scoring, and affinity maturation.
