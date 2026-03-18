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

## Full Pipeline Order (pipeline.py)
1. `binder_gen` – structure sampling via `module.run_pipeline(args)`
2. `inverse_folding` – fixed-positions CSV + ProteinMPNN + fasta_to_csv + graft_sequences
3. `run_fampnn` – sidechain packing on grafted PDBs
4. `run_af3score` – AF3Score scoring + best_models selection
5. `run_fastrelax_interface` – PyRosetta interface FastRelax on best_models; maturation.csv + partial_flow_ready/ written to af3score/
6. `run_partial_flow` – partial flow refinement per family using merged maturation residues

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
| `_detect_designed_chains(output_dir)` | Returns chain IDs that have both B-factor 4.0 and 2.0 (backbone PDBs only — FAMPNN overwrites B-factors with PSCE) |
| `run_protein_mpnn(output_dir, csv_path, chain_list, cfg)` | Calls `protein_mpnn_run.main()` directly; returns list of seqs dirs; supports hydrophobic bias split |
| `mpnn_fasta_to_csv(input_dirs, output_csv, suffix, top_n=5)` | Reads FASTA files from MPNN seqs dir(s); deduplicates; keeps top 5 lowest-score sequences per design; writes `seqsfinal_result.csv` with columns `link_name, seq, seq_idx, score` |
| `graft_sequences_to_pdbs(output_dir, csv_path, designed_chains)` | For each row in `seqsfinal_result.csv`, replaces residue names on designed chains in the backbone PDB with the MPNN sequence; writes `mpnn_output/<basename>_<seq_idx>.pdb` |
| `pack_sidechains_dir(input_dir, output_dir, checkpoint)` | Runs FAMPNN sidechain packing on every PDB in input_dir; writes full-atom PDBs to output_dir with PSCE confidence in B-factors |
| `_fixed_motif_for_pdb(pdb_path, maturation_residues, binder_chain, original_fixed_resnums)` | Returns motif_index string for one partial flow backbone: union of B-factor-1.0 framework, maturation residues, and original fixed positions |
| `write_partial_flow_fixed_positions_csv(rows, out_path)` | Writes pre-built rows to `partial_flow_mpnn_fixed_positions.csv` (tab-separated, same format as `mpnn_fixed_positions.csv`) |

## Pipeline Steps & State File
Seven tracked steps written to `<output_dir>/pipeline_state.json`:
1. `binder_gen` – structure sampling
2. `fixed_positions_csv` – `mpnn_fixed_positions.csv` creation
3. `protein_mpnn` – ProteinMPNN inverse folding + fasta_to_csv + graft_sequences (all in one state block)
4. `fampnn` – FAMPNN sidechain packing → `fampnn_designs/<name>_<seq_idx>.pdb`
5. `af3score` – AF3Score scoring → `af3score/af3score_metrics.csv`; best models (ipTM > first_round_iptm, top num_samples) → `af3score/best_models/`
6. `fastrelax` – PyRosetta interface FastRelax → `af3score/best_models_relaxed/`; maturation candidates (REU < cutoff) → `af3score/maturation.csv`; family merge → `af3score/partial_flow_ready/`
7. `partial_flow` – partial flow backbone gen per family + MPNN (T=0.1) on all dump backbones → `partial_flow/dump/mpnn_output/`

## Output Directory Layout
```
<output_dir>/
├── *.pdb                        # backbone PDBs (B-factor encoding intact)
├── pipeline_state.json
├── mpnn_fixed_positions.csv
├── mpnn_output/
│   ├── seqs/                    # FASTA files (normal + biased appended)
│   ├── hydrophobic_bias.jsonl   # written when mpnn_hydrophobic_bias: true
│   ├── seqsfinal_result.csv
│   └── <name>_<seq_idx>.pdb    # grafted backbone PDBs
├── fampnn_designs/
│   └── <name>_<seq_idx>.pdb    # full-atom PDBs (B-factor = PSCE confidence)
├── af3score/
│   ├── af3score_metrics.csv
│   ├── best_models/             # PDBs with ipTM > first_round_iptm (top num_samples)
│   ├── best_models_relaxed/     # FastRelax output PDBs
│   ├── maturation.csv           # columns: binder_name, proteinmpnn_family, path, maturation_residues
│   └── partial_flow_ready/
│       ├── merged_residues.csv  # columns: family, path, merged_residues
│       └── <family>.pdb         # representative PDB per family (copy of best_models_relaxed/)
└── partial_flow/
    ├── partial_flow_mpnn_fixed_positions.csv  # merged framework+maturation+original fixed positions
    ├── <family>/                # per-family partial flow backbone PDBs + input/
    └── dump/
        ├── <name>_<fam>_<pf>.pdb              # backbone PDBs (3-part name)
        └── mpnn_output/
            ├── seqs/
            ├── seqsfinal_result.csv
            └── <name>_<fam>_<pf>_<seq>.pdb   # final grafted designs (4-part name)
```

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
- `batch_size` must not exceed `num_seqs_per_target` (or the per-run half when `mpnn_hydrophobic_bias` is on) — raises `ValueError` if violated.

## Hydrophobic Bias (ProteinMPNN)
- Enable with `mpnn_hydrophobic_bias: true` in YAML.
- Splits `num_seqs_per_target` evenly: `n//2` normal, `n - n//2` biased.
- Biased run uses `bias_AA_jsonl` with `{W: 1.5, F: 1.5, Y: 1.0, L: 1.0, I: 1.0}` to favor large hydrophobic residues at the interface.
- Both runs output to the same `mpnn_output/seqs/` — biased FASTAs are appended to normal ones so numbering is continuous.
- **Constraint**: `batch_size` must be ≤ the per-run sequence count (`num_seqs_per_target // 2` when bias is on).

## AF3Score
- Separate package at `/home/seanwang/af3score`; install with `pip install -e /path/to/af3score`
- Runs after FAMPNN; requires `fampnn_designs/` to exist with PDBs (no fallback)
- Invoked via subprocess with `cwd=af3score_dir` so internal relative script paths resolve; input/output paths are made absolute before passing
- Requires its own Python env if JAX conflicts with PPIFlow env → use `af3score_python` key
- After scoring: filters `iptm > first_round_iptm`, takes top `num_samples`, copies PDBs to `af3score/best_models/`
- `run_af3score(output_dir, cfg, state, num_samples)` in `pipeline.py`

## FastRelax (PyRosetta)
- Script: `pyrosetta_scripts/interface_fastrelax.py`
- Mirrors `demo_scripts/interface_analysis/codes/native.xml`: neighborhood selector (20 Å around binder interface residues), restrict-to-repacking task ops, FastRelax lbfgs 2 repeats, MoveMap binder bb+chi / target chi-only
- Interface residues detected via `biopython_utils.hotspot_residues()` (4 Å atom cutoff) → converted to Rosetta pose indices for `ResidueIndexSelector`
- Post-relax: per-residue interface REU via energy graph; residues with REU < `reu_cutoff` (default -1.0) → `af3score/maturation.csv` as `{binder_name: [resnum, ...]}`
- Chain detection: binder from `_detect_designed_chains(output_dir)` (backbone B-factors); target from PDB chain list minus designed chains
- Each PDB runs in a **spawned** worker process (not fork — PyRosetta is not fork-safe)
- Requires separate Python env with PyRosetta → `fastrelax_python` key

## YAML Keys for AF3Score (all `pipeline_*.yaml`)
| Key | Required | Default |
|---|---|---|
| `af3score_dir` | Yes (omit to skip) | – |
| `af3score_weights` | Yes (when af3score_dir is set) | – |
| `af3score_python` | No | `sys.executable` |
| `af3score_num_workers` | No | 4 |
| `af3score_db_dir` | No (string or list) | – |
| `first_round_iptm` | No (omit to skip best_models selection) | – |

## YAML Keys for ProteinMPNN (all `pipeline_*.yaml`)
| Key | Required | Default |
|---|---|---|
| `mpnn_weights` | Yes (omit to skip MPNN) | – |
| `model_name` | No | `v_48_020` |
| `num_seqs_per_target` | No | `8` |
| `batch_size` | No | `1` |
| `sampling_temp` | No | `"0.1"` |
| `mpnn_omit_AAs` | No | `"X"` |
| `mpnn_hydrophobic_bias` | No | `false` |

## YAML Keys for FastRelax (all `pipeline_*.yaml`)
| Key | Required | Default |
|---|---|---|
| `fastrelax_python` | Yes (omit to skip) | – |
| `fastrelax_cpus` | Yes (omit to skip) | – |
| `fastrelax_score_cutoff` | No | `5000000.0` |
| `fastrelax_reu_cutoff` | No | `-1.0` |

## Partial Flow (run_partial_flow)
Two-pass implementation:

**Pass 1 – backbone generation (per family):**
- Reads `af3score/partial_flow_ready/merged_residues.csv`.
- For each family: formats maturation residues as `"A45,A67,A92"`, calls `sample_antibody_nanobody_partial` or `sample_binder_partial`.
- Backbone PDBs saved to `partial_flow/<family>/sample{n}.pdb`.
- Each backbone is copied to `partial_flow/dump/{name}_{fam}_{pf}.pdb` (3-part name).
- Partial flow B-factor encoding: binder 1.0 = framework, 0.0 = CDR; antigen 2.0 = hotspot, 0.0 = other.

**Pass 2 – ProteinMPNN on all dump backbones (once, after all families):**
- Builds `partial_flow/partial_flow_mpnn_fixed_positions.csv` — one row per dump backbone.
- Each row's fixed positions = union of:
  1. B-factor 1.0 residues on binder chain (partial flow framework)
  2. Maturation residues for the family (don't redesign good contacts)
  3. Original fixed positions from `mpnn_fixed_positions.csv` (framework from first-round MPNN; keyed by backbone name = family name directly — no rsplit)
- Runs MPNN at **T=0.1** (hardcoded, conservative) on `partial_flow/dump/`.
- Grafted designs → `partial_flow/dump/mpnn_output/{name}_{fam}_{pf}_{seq}.pdb` (4-part name).
- `retry_Limit` defaults to 10 if not set in YAML.
- State key: `partial_flow`.

## YAML Keys for Partial Flow (all `pipeline_*.yaml`)
| Key | Required | Default |
|---|---|---|
| `partial_flow_start_t` | No (omit to skip) | `0.6` when key present but null |
| `retry_Limit` | No | `10` |

## B-Factor Encoding in Output PDBs
| Value | Meaning |
|---|---|
| 4.0 | Antibody/nanobody framework residues (fixed) |
| 2.0 | CDR residues (designed) |
| 1.0 | Antigen hotspot residues |
| 0.0 | Antigen non-hotspot / binder chain (binder task) |

Note: in binder outputs, `hotspot_mask + target_interface_mask` can sum to 2.0 on the antigen chain — designed-chain detection therefore requires **both** 4.0 AND 2.0 to be present on the same chain.
FAMPNN overwrites B-factors with PSCE confidence — always use backbone PDBs in `output_dir` for chain detection, never `fampnn_designs/` or `af3score/`.

**Partial flow backbone PDBs** use a different encoding (from `flow_module_antibody_partial.py`):
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
| `pipeline_*.yaml` | Templates for `pipeline.py` (include MPNN/FAMPNN/AF3Score/FastRelax blocks) |

## PyRosetta Scripts (`pyrosetta_scripts/`)
| File | Purpose |
|---|---|
| `interface_fastrelax.py` | Parallel interface FastRelax + maturation residue analysis |
| `biopython_utils.py` | `hotspot_residues()`, `secondary_structure()` |
| `pyrosetta_utils.py` | `fastrelax()`, `energy_interacting_residues()` — has module-level `pr.init()`, do not import in main process |

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

## Residue Interaction Guidance
See `CLAUDE_RL.md`.
