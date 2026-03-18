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
import glob
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
    _fixed_motif_for_pdb,
    write_partial_flow_fixed_positions_csv,
    run_protein_mpnn,
)

from helper_functions import *

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

    seqs_csv = os.path.join(output_dir, "mpnn_output", "seqsfinal_result.csv")

    if state.is_done("protein_mpnn"):
        print("[pipeline.py] Skipping protein_mpnn (already done).")
    else:
        seqs_dirs = run_protein_mpnn(output_dir, csv_path, chain_list, cfg)
        mpnn_fasta_to_csv(
            input_dirs=seqs_dirs,
            output_csv=seqs_csv,
            suffix=".pdb",
        )
        graft_sequences_to_pdbs(
            output_dir=output_dir,
            csv_path=seqs_csv,
            designed_chains=designed_chains if designed_chains else chain_list.split(),
        )
        state.mark_done("protein_mpnn")

def run_fampnn(output_dir: str, cfg: dict, state: PipelineState) -> None:
    """Pack sidechains on grafted PDBs using FAMPNN."""
    if not cfg.get("fampnn_weights"):
        print("[pipeline.py] 'fampnn_weights' not set – skipping FAMPNN sidechain packing.")
        return

    if state.is_done("fampnn"):
        print("[pipeline.py] Skipping fampnn (already done).")
        return

    pack_sidechains_dir(
        input_dir=os.path.join(output_dir, "mpnn_output"),
        output_dir=os.path.join(output_dir, "fampnn_designs"),
        checkpoint=cfg["fampnn_weights"],
    )
    state.mark_done("fampnn")

def run_af3score(output_dir: str, cfg: dict, state: PipelineState, num_samples: int = 0) -> None:
    """Run AF3Score on FAMPNN designs (or backbone PDBs if FAMPNN was skipped).

    Required YAML keys:
        af3score_dir     – path to the af3score repo (must contain af3score_pipeline.py)
        af3score_weights – path to the AlphaFold3 model weights file

    Optional YAML keys:
        af3score_python      – Python executable to use (default: sys.executable)
        af3score_num_workers – worker count for preprocessing (default: 4)
        af3score_db_dir      – AF3 database dir(s); string or list of strings
    """
    if not cfg.get("af3score_dir"):
        print("[pipeline.py] 'af3score_dir' not set – skipping AF3Score.")
        return
    if not cfg.get("af3score_weights"):
        print("[pipeline.py] 'af3score_weights' not set – skipping AF3Score.")
        return

    if state.is_done("af3score"):
        print("[pipeline.py] Skipping af3score (already done).")
        return

    fampnn_dir = os.path.abspath(os.path.join(output_dir, "fampnn_designs"))
    if not os.path.isdir(fampnn_dir) or not any(
        f.endswith(".pdb") for f in os.listdir(fampnn_dir)
    ):
        raise FileNotFoundError(
            f"AF3Score requires FAMPNN output but no PDBs found in {fampnn_dir}. "
            "Set 'fampnn_weights' in your config and ensure FAMPNN completes first."
        )

    af3score_output = os.path.abspath(os.path.join(output_dir, "af3score"))
    af3score_dir = os.path.expanduser(cfg["af3score_dir"])
    python_exec = cfg.get("af3score_python") or sys.executable

    cmd = [
        python_exec,
        "af3score_pipeline.py",
        "--input", fampnn_dir,
        "--output_dir", af3score_output,
        "--weights", cfg["af3score_weights"],
    ]

    if cfg.get("af3score_num_workers"):
        cmd += ["--num_workers", str(cfg["af3score_num_workers"])]

    db_dirs = cfg.get("af3score_db_dir", [])
    if isinstance(db_dirs, str):
        db_dirs = [db_dirs]
    for db in db_dirs:
        cmd += ["--db_dir", db]

    print(f"\n[pipeline.py] Running AF3Score")
    print(f"[pipeline.py]   input       : {fampnn_dir}")
    print(f"[pipeline.py]   output      : {af3score_output}")
    print(f"[pipeline.py]   cwd         : {af3score_dir}")
    print(f"[pipeline.py]   python      : {python_exec}\n")

    import subprocess
    subprocess.run(cmd, check=True, cwd=af3score_dir)
    state.mark_done("af3score")

    # ── Select best models by ipTM ────────────────────────────────────────────
    first_round_iptm = cfg.get("first_round_iptm")
    if first_round_iptm is None:
        print("[pipeline.py] 'first_round_iptm' not set – skipping best-model selection.")
        return

    import csv, shutil

    metric_csv = os.path.join(af3score_output, "af3score_metrics.csv")
    if not os.path.isfile(metric_csv):
        print(f"[pipeline.py] Warning: metrics CSV not found at {metric_csv} – skipping best-model selection.")
        return

    # Read CSV, filter ipTM > threshold, sort descending, cap at num_samples
    passing = []
    with open(metric_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                iptm_val = float(row["iptm"])
            except (KeyError, ValueError):
                continue
            if iptm_val > float(first_round_iptm):
                passing.append((iptm_val, row["description"]))

    passing.sort(key=lambda x: x[0], reverse=True)

    top = passing[:num_samples] if num_samples > 0 else passing

    best_dir = os.path.join(af3score_output, "best_models")
    os.makedirs(best_dir, exist_ok=True)

    copied = 0
    for iptm_val, desc in top:
        src = os.path.join(fampnn_dir, f"{desc}.pdb")
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(best_dir, f"{desc}.pdb"))
            copied += 1
        else:
            print(f"[pipeline.py] Warning: PDB not found for {desc} – skipping.")

    print(f"\n[pipeline.py] Best models: {copied} PDBs copied to {best_dir}")
    print(f"[pipeline.py]   threshold : ipTM > {first_round_iptm}")
    print(f"[pipeline.py]   passing   : {len(passing)} total, kept top {len(top)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# FastRelax step
# ---------------------------------------------------------------------------

def run_fastrelax_interface(output_dir: str, cfg: dict, state: PipelineState) -> None:
    """Run PyRosetta interface FastRelax on the best AF3Score models.

    Reads PDBs from <output_dir>/af3score/best_models/ and writes relaxed
    structures to <output_dir>/fastrelax_designs/.

    Required YAML keys:
        fastrelax_cpus    – number of parallel worker processes (omit to skip)
        fastrelax_python  – Python executable with PyRosetta installed

    Optional YAML keys:
        binder_chain / heavy_chain   – chain ID of the designed binder (default: A)
        fastrelax_score_cutoff       – total score cutoff (default: 5e6)
    """
    if not cfg.get("fastrelax_cpus"):
        print("[pipeline.py] 'fastrelax_cpus' not set – skipping interface FastRelax.")
        return
    if not cfg.get("fastrelax_python"):
        print("[pipeline.py] 'fastrelax_python' not set – skipping interface FastRelax.")
        return

    if state.is_done("fastrelax"):
        print("[pipeline.py] Skipping fastrelax (already done).")
        return

    best_models_dir = os.path.abspath(os.path.join(output_dir, "af3score", "best_models"))
    if not os.path.isdir(best_models_dir) or not any(
        f.endswith(".pdb") for f in os.listdir(best_models_dir)
    ):
        raise FileNotFoundError(
            f"FastRelax requires AF3Score best_models but no PDBs found in {best_models_dir}. "
            "Ensure af3score + first_round_iptm are configured and run first."
        )

    fastrelax_out = os.path.abspath(os.path.join(output_dir, "af3score", "best_models_relaxed"))

    # -- Detect binder chain from backbone B-factors (reliable source) -------
    # Backbone PDBs in output_dir still carry the original B-factor encoding
    # (4.0 = framework, 2.0 = CDR).  FAMPNN overwrites B-factors with PSCE
    # confidence, so we must NOT read from fampnn_designs/ or best_models/.
    designed_chains = _detect_designed_chains(output_dir)
    if designed_chains:
        # Nanobody / antibody: heavy chain is first designed chain
        binder_chain = designed_chains[0]
    else:
        # Binder task: no 4.0/2.0 encoding; fall back to config key
        binder_chain = cfg.get("binder_chain") or "A"

    # -- Detect target chain from actual PDB chain list ----------------------
    # Read chain IDs from the first PDB in best_models and subtract the binder.
    first_pdb = sorted(
        f for f in os.listdir(best_models_dir) if f.endswith(".pdb")
    )[0]
    all_chains = []
    with open(os.path.join(best_models_dir, first_pdb)) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[21] not in all_chains:
                all_chains.append(line[21])

    target_chains = [c for c in all_chains if c not in (designed_chains or [binder_chain])]
    if not target_chains:
        raise ValueError(
            f"Could not detect target chain in {first_pdb}. "
            f"Chains found: {all_chains}, binder: {binder_chain}"
        )
    target_chain = target_chains[0]

    script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "pyrosetta_scripts", "interface_fastrelax.py")
    )
    python_exec = cfg["fastrelax_python"]
    cpus = int(cfg["fastrelax_cpus"])

    maturation_csv = os.path.abspath(os.path.join(output_dir, "af3score", "maturation.csv"))

    cmd = [
        python_exec, script_path,
        "--input_dir",      best_models_dir,
        "--output_dir",     fastrelax_out,
        "--maturation_csv", maturation_csv,
        "--binder_chain",   binder_chain,
        "--target_chain",   target_chain,
        "--cpus",           str(cpus),
    ]
    if cfg.get("fastrelax_score_cutoff"):
        cmd += ["--score_cutoff", str(cfg["fastrelax_score_cutoff"])]
    if cfg.get("fastrelax_reu_cutoff"):
        cmd += ["--reu_cutoff", str(cfg["fastrelax_reu_cutoff"])]

    print(f"\n[pipeline.py] Running interface FastRelax")
    print(f"[pipeline.py]   input         : {best_models_dir}")
    print(f"[pipeline.py]   output        : {fastrelax_out}")
    print(f"[pipeline.py]   maturation csv: {maturation_csv}")
    print(f"[pipeline.py]   binder        : chain {binder_chain}")
    print(f"[pipeline.py]   target        : chain {target_chain}")
    print(f"[pipeline.py]   cpus          : {cpus}")
    print(f"[pipeline.py]   python        : {python_exec}\n")

    import subprocess
    subprocess.run(cmd, check=True)
    state.mark_done("fastrelax")


# ---------------------------------------------------------------------------
# Partial flow refinement
# ---------------------------------------------------------------------------

def run_partial_flow(output_dir: str, cfg: dict, state: PipelineState, num_samples: int) -> None:
    """Run partial flow refinement for each family in partial_flow_ready/merged_residues.csv,
    then run ProteinMPNN on the resulting backbones (with framework + maturation residues fixed),
    graft sequences, and collect all designs into partial_flow/dump/ with naming:
        <name>_<family_num>_<pf_idx>_<mpnn_idx>.pdb

    Required YAML key:
        partial_flow_start_t  – start_t for partial flow (omit to skip); default 0.6 when key present but null

    Output:
        partial_flow/<family>/  – per-family backbone PDBs + MPNN output
        partial_flow/dump/      – final named PDBs for all families
    """
    if "partial_flow_start_t" not in cfg:
        print("[pipeline.py] 'partial_flow_start_t' not set – skipping partial flow.")
        return
    start_t = cfg["partial_flow_start_t"] if cfg["partial_flow_start_t"] is not None else 0.6

    if state.is_done("partial_flow"):
        print("[pipeline.py] Skipping partial_flow (already done).")
        return

    merged_csv = os.path.abspath(
        os.path.join(output_dir, "af3score", "partial_flow_ready", "merged_residues.csv")
    )
    if not os.path.isfile(merged_csv):
        raise FileNotFoundError(
            f"Partial flow requires {merged_csv}. "
            "Ensure fastrelax + first_round_iptm are configured and ran first."
        )

    # Determine partial flow module from original task
    task = cfg.get("task", "").lower()
    if "binder" in task:
        module = sample_binder_partial
        partial_task = "partial_flow_binder"
    elif "nanobody" in task:
        module = sample_antibody_nanobody_partial
        partial_task = "partial_flow_nanobody"
    else:
        module = sample_antibody_nanobody_partial
        partial_task = "partial_flow_antibody"

    # Binder chain for residue spec formatting and MPNN
    designed_chains = _detect_designed_chains(output_dir)
    binder_chain = designed_chains[0] if designed_chains else cfg.get("binder_chain", "A")

    import csv as _csv, ast as _ast

    with open(merged_csv, newline="") as fh:
        families = list(_csv.DictReader(fh))

    if not families:
        print("[pipeline.py] merged_residues.csv is empty – nothing to run.")
        return

    cfg_name  = cfg.get("name", "design")
    pf_dir    = os.path.abspath(os.path.join(output_dir, "partial_flow"))
    dump_dir  = os.path.join(pf_dir, "dump")
    os.makedirs(dump_dir, exist_ok=True)

    # ── Load original mpnn_fixed_positions.csv → per-family fixed residue sets ─
    # Maps family name (e.g. "nanobody_3") → set of int residue numbers that were
    # fixed in the first-round MPNN run.  These are merged into the partial flow
    # fixed positions so framework residues stay frozen in both passes.
    orig_family_fixed: dict = {}
    orig_fixed_csv = os.path.join(output_dir, "mpnn_fixed_positions.csv")
    if os.path.isfile(orig_fixed_csv):
        import csv as _csv_orig
        with open(orig_fixed_csv, newline="") as fh:
            for row_orig in _csv_orig.DictReader(fh, delimiter="\t"):
                # pdb_name is the backbone name (e.g. "nanobody_3") which IS the family name.
                # Do not rsplit — the backbone name maps directly to the family key.
                pdb_name_orig = str(row_orig["pdb_name"])
                motif = str(row_orig.get("motif_index", ""))
                resnums: set = set()
                for token in motif.replace("-", " ").split():
                    try:
                        resnums.add(int(token))
                    except ValueError:
                        pass
                orig_family_fixed.setdefault(pdb_name_orig, set()).update(resnums)
    else:
        print(f"[pipeline.py] WARNING: {orig_fixed_csv} not found – original fixed positions will not be merged.")

    print(f"\n[pipeline.py] Partial flow: {len(families)} families, start_t={start_t}")

    # ── Pass 1: run partial flow per family; copy backbones to dump/ ─────────
    # Accumulate fixed-position rows as we go so we can write one CSV afterward.
    fixed_rows = []   # {"pdb_name": dump_basename, "motif_index": str}

    for row in families:
        family   = row["family"]
        pdb_path = os.path.abspath(row["path"])
        residues = _ast.literal_eval(row["merged_residues"])

        if not residues:
            print(f"[pipeline.py] Family {family}: no residues – skipping.")
            continue

        residue_spec = ",".join(f"{binder_chain}{r}" for r in sorted(residues))
        family_output = os.path.abspath(os.path.join(pf_dir, family))
        os.makedirs(family_output, exist_ok=True)

        family_cfg = dict(cfg)
        family_cfg["start_t"]         = float(start_t)
        family_cfg["fixed_positions"] = residue_spec
        family_cfg["name"]            = family

        if partial_task == "partial_flow_binder":
            family_cfg["input_pdb"] = pdb_path
        else:
            family_cfg["complex_pdb"]  = pdb_path
            family_cfg["cdr_position"] = residue_spec

        print(f"[pipeline.py]   Backbone: {family}: {residue_spec} → {family_output}")
        args = TASK_MAP[partial_task][1](family_cfg, family_output, num_samples)
        module.run_pipeline(args)

        # Copy backbones to dump/ with 3-part name; build fixed-position rows
        family_num    = family.rsplit("_", 1)[-1]
        pf_backbones  = sorted(glob.glob(os.path.join(family_output, "sample*.pdb")))
        if not pf_backbones:
            print(f"[pipeline.py] WARNING: no backbone PDBs in {family_output} – skipping MPNN for {family}.")
            continue

        for bb_path in pf_backbones:
            pf_idx       = os.path.splitext(os.path.basename(bb_path))[0].replace("sample", "")
            dump_basename = f"{cfg_name}_{family_num}_{pf_idx}"
            dst           = os.path.join(dump_dir, f"{dump_basename}.pdb")
            shutil.copy2(bb_path, dst)

            orig_fixed = orig_family_fixed.get(family, set())
            motif_index = _fixed_motif_for_pdb(dst, residues, binder_chain, orig_fixed)
            fixed_rows.append({"pdb_name": dump_basename, "motif_index": motif_index})

    if not fixed_rows:
        print("[pipeline.py] No partial flow backbones produced – skipping MPNN.")
        state.mark_done("partial_flow")
        return

    # ── Pass 2: write combined fixed-positions CSV at partial_flow/ level ────
    pf_fixed_csv = os.path.join(pf_dir, "partial_flow_mpnn_fixed_positions.csv")
    write_partial_flow_fixed_positions_csv(fixed_rows, pf_fixed_csv)

    # ── Pass 3: run MPNN once on all dump/ backbones ──────────────────────────
    # Use T=0.1 for partial flow MPNN (lower temperature → more conservative sequence design)
    pf_mpnn_cfg = dict(cfg)
    pf_mpnn_cfg["sampling_temp"] = "0.1"

    print(f"\n[pipeline.py]   MPNN: {len(fixed_rows)} backbones in {dump_dir} (T=0.1)")
    pf_seqs_dirs = run_protein_mpnn(dump_dir, pf_fixed_csv, binder_chain, pf_mpnn_cfg)

    pf_seqs_csv = os.path.join(dump_dir, "mpnn_output", "seqsfinal_result.csv")
    mpnn_fasta_to_csv(input_dirs=pf_seqs_dirs, output_csv=pf_seqs_csv, suffix=".pdb")

    graft_sequences_to_pdbs(
        output_dir=dump_dir,
        csv_path=pf_seqs_csv,
        designed_chains=[binder_chain],
    )
    # Grafted PDBs land in dump/mpnn_output/{cfg_name}_{fam}_{pf}_{seq}.pdb
    # (graft_sequences_to_pdbs already names them {link_base}_{seq_idx}.pdb)

    print(f"\n[pipeline.py] Partial flow designs → {dump_dir}/mpnn_output/")
    state.mark_done("partial_flow")


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
    run_fampnn(cli_args.output, cfg, state)
    run_af3score(cli_args.output, cfg, state, num_samples=cli_args.num_samples)
    run_fastrelax_interface(cli_args.output, cfg, state)
    run_partial_flow(cli_args.output, cfg, state, num_samples=cli_args.num_samples)


if __name__ == "__main__":
    main()
