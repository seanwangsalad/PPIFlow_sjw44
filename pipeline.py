"""
PPIFlow Unified Pipeline
========================
Run any PPIFlow task by pointing at a single YAML file instead of passing
many command-line flags.

Usage:
    python pipeline.py --config my_run.yaml --output /path/to/out --num_samples 5

The YAML file must contain a ``task`` field that selects the mode:

    binder                  – protein binder design
    antibody                – antibody CDR design (heavy + light chain)
    nanobody                – nanobody CDR design (heavy chain only)
    monomer                 – unconditional monomer generation
    motif_scaffolding       – motif-based scaffolding
    partial_flow_antibody   – partial-flow refinement for antibody
    partial_flow_nanobody   – partial-flow refinement for nanobody
    partial_flow_binder     – partial-flow refinement for binder

--output and --num_samples are intentionally CLI-only so the same YAML can be
reused across runs with different output locations or sample counts.

See configs/pipeline_*.yaml for task-specific templates.
"""

import argparse
import os, shutil
import sys
import yaml
from helper_functions import (
    PipelineState,
    _build_binder_args,
    _build_antibody_nanobody_args,
    _build_monomer_args,
    _build_partial_antibody_nanobody_args,
    _build_partial_binder_args,
    create_mpnn_fixed_positions_csv,
    _detect_designed_chains,
    run_protein_mpnn,
)

import scripts.sample_binder as sample_binder
import scripts.sample_antibody_nanobody as sample_antibody_nanobody
import scripts.sample_monomer as sample_monomer
import scripts.sample_antibody_nanobody_partial as sample_antibody_nanobody_partial
import scripts.sample_binder_partial as sample_binder_partial


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def load_pipeline_config(yaml_path: str) -> dict:
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Pipeline config not found: {yaml_path}")
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    if "task" not in cfg:
        raise ValueError("Pipeline config must contain a 'task' field.")
    return cfg


# ---------------------------------------------------------------------------
# Task dispatch
# ---------------------------------------------------------------------------

TASK_MAP = {
    "binder":                  (sample_binder,                    _build_binder_args),
    "antibody":                (sample_antibody_nanobody,         _build_antibody_nanobody_args),
    "nanobody":                (sample_antibody_nanobody,         _build_antibody_nanobody_args),
    "monomer":                 (sample_monomer,                   _build_monomer_args),
    "motif_scaffolding":       (sample_monomer,                   _build_monomer_args),
    "partial_flow_antibody":   (sample_antibody_nanobody_partial, _build_partial_antibody_nanobody_args),
    "partial_flow_nanobody":   (sample_antibody_nanobody_partial, _build_partial_antibody_nanobody_args),
    "partial_flow_binder":     (sample_binder_partial,            _build_partial_binder_args),
}


def binder_gen(yaml_path: str, output_dir: str, num_samples: int) -> None:
    cfg = load_pipeline_config(yaml_path)
    task = cfg["task"].strip().lower()

    if task not in TASK_MAP:
        raise ValueError(
            f"Unknown task '{task}'. Valid options: {', '.join(TASK_MAP)}"
        )

    module, args_builder = TASK_MAP[task]

    args = args_builder(cfg, output_dir, num_samples)

    if hasattr(module, "validate_inputs"):
        module.validate_inputs(args)

    print(f"\n[pipeline.py] Task       : {task}")
    print(f"[pipeline.py] Module     : {module.__name__}")
    print(f"[pipeline.py] Output dir : {args.output_dir}")
    print(f"[pipeline.py] Num samples: {num_samples}\n")

    module.run_pipeline(args)


def inverse_folding(output_dir: str, cfg: dict, state: PipelineState) -> None:
    """Create the fixed-positions CSV then run ProteinMPNN inverse folding."""
    if not cfg.get("mpnn_weights"):
        print("[pipeline.py] 'mpnn_weights' not set in config – skipping inverse folding.")
        return

    print(f"\n[pipeline.py] Inverse folding – output dir: {output_dir}")

    if state.is_done("fixed_positions_csv"):
        csv_path = os.path.join(output_dir, "mpnn_fixed_positions.csv")
        print("[pipeline.py] Skipping fixed_positions_csv (already done).")
    else:
        csv_path = create_mpnn_fixed_positions_csv(output_dir)
        state.mark_done("fixed_positions_csv")

    # Antibody/nanobody: chains with B-factor 4.0 (framework) AND 2.0 (CDR)
    # Binder: no such chains → fall back to binder_chain from config
    designed_chains = _detect_designed_chains(output_dir)
    if designed_chains:
        chain_list = " ".join(designed_chains)
    else:
        chain_list = cfg.get("binder_chain", "")
        if not chain_list:
            print("[pipeline.py] Warning: could not determine designed chain – skipping ProteinMPNN.")
            return

    print(f"[pipeline.py] Designing chain(s): {chain_list!r}")

    if state.is_done("protein_mpnn"):
        print("[pipeline.py] Skipping protein_mpnn (already done).")
    else:
        run_protein_mpnn(output_dir, csv_path, chain_list, cfg)
        state.mark_done("protein_mpnn")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PPIFlow unified pipeline – run any task from a YAML file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the pipeline YAML configuration file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for results.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to generate (default: 5).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume a previous run, skipping already-completed steps.",
    )
    cli_args = parser.parse_args()
    cfg = load_pipeline_config(cli_args.config)
    state = PipelineState(cli_args.output, cfg, resume=cli_args.resume)

    if state.is_done("binder_gen"):
        print("[pipeline.py] Skipping binder_gen (already done).")
    else:
        binder_gen(cli_args.config, cli_args.output, cli_args.num_samples)
        state.mark_done("binder_gen")

    inverse_folding(cli_args.output, cfg, state)


if __name__ == "__main__":
    main()
