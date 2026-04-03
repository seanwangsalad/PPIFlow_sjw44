"""
PPIFlow Unified Pipeline
========================
Run any PPIFlow generation task by pointing at a single YAML file.

Usage:
    python pipeline.py --config my_run.yaml --output /path/to/out --num_samples 5

The YAML file must contain a ``task`` field:

    binder              – protein binder design
    antibody            – antibody CDR design (heavy + light chain)
    nanobody            – nanobody CDR design (heavy chain only)
    monomer             – unconditional monomer generation
    motif_scaffolding   – motif-based scaffolding
    partial_flow_ab     – partial-flow refinement for antibody or nanobody
    partial_flow_binder – partial-flow refinement for binder

--output and --num_samples are CLI-only so the same YAML can be reused
across runs with different output locations or sample counts.

See configs/pipeline_*.yaml for task-specific templates.
"""

import argparse
import os
import sys
import yaml

from helper_functions import (
    PipelineState,
    _build_binder_args,
    _build_antibody_nanobody_args,
    _build_monomer_args,
    _build_partial_antibody_nanobody_args,
    _build_partial_binder_args,
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
    "binder":               (sample_binder,                    _build_binder_args),
    "antibody":             (sample_antibody_nanobody,         _build_antibody_nanobody_args),
    "nanobody":             (sample_antibody_nanobody,         _build_antibody_nanobody_args),
    "monomer":              (sample_monomer,                   _build_monomer_args),
    "motif_scaffolding":    (sample_monomer,                   _build_monomer_args),
    "partial_flow_ab":      (sample_antibody_nanobody_partial, _build_partial_antibody_nanobody_args),
    "partial_flow_binder":  (sample_binder_partial,            _build_partial_binder_args),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PPIFlow unified pipeline – backbone generation."
    )
    parser.add_argument("--config",      type=str, required=True,
                        help="Path to the pipeline YAML configuration file.")
    parser.add_argument("--output",      type=str, required=True,
                        help="Output directory for results.")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to generate (default: 5).")
    parser.add_argument("--resume",      action="store_true", default=False,
                        help="Resume a previous run, skipping already-completed steps.")
    cli_args = parser.parse_args()

    cfg   = load_pipeline_config(cli_args.config)
    task  = cfg["task"].strip().lower()
    state = PipelineState(cli_args.output, cfg, resume=cli_args.resume)

    if task not in TASK_MAP:
        raise ValueError(f"Unknown task '{task}'. Valid options: {', '.join(TASK_MAP)}")

    if state.is_done("binder_gen"):
        print("[pipeline.py] Skipping binder_gen (already done).")
        return

    module, args_builder = TASK_MAP[task]
    args = args_builder(cfg, cli_args.output, cli_args.num_samples)

    if hasattr(module, "validate_inputs"):
        module.validate_inputs(args)

    print(f"\n[pipeline.py] Task       : {task}")
    print(f"[pipeline.py] Module     : {module.__name__}")
    print(f"[pipeline.py] Output dir : {args.output_dir}")
    print(f"[pipeline.py] Num samples: {cli_args.num_samples}\n")

    module.run_pipeline(args)
    state.mark_done("binder_gen")


if __name__ == "__main__":
    main()
