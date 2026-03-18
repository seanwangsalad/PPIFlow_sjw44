#!/usr/bin/env python3
"""
interface_fastrelax.py
======================
Parallel interface FastRelax using PyRosetta, followed by per-residue
interface energy analysis for affinity maturation candidate identification.

Protocol mirrors demo_scripts/interface_analysis/codes/native.xml:
  - Interface residues identified with biopython_utils.hotspot_residues()
    (binder atoms within 4 Å of target atoms)
  - Neighborhood selector: 20 Å around those interface residues
  - OperateOnResidueSubset: PreventRepacking on all non-neighborhood residues
  - RestrictToRepacking on the rest (no sequence design)
  - FastRelax: lbfgs_armijo_nonmonotone, 2 repeats, max_iter 200
  - MoveMap: binder chain → bb + chi free; target chain(s) → chi only, bb frozen
  - ScoreCutoffFilter: discard poses above total_score cutoff (default 5e6)

After FastRelax, per-residue interface REU scores are computed via the Rosetta
energy graph. Residues with REU < reu_cutoff (default -5) are written to
maturation.csv in the af3score output directory.

Each PDB is processed in a separate spawned worker process.

Usage (standalone):
    python interface_fastrelax.py \\
        --input_dir       af3score/best_models \\
        --output_dir      fastrelax_designs \\
        --maturation_csv  af3score/maturation.csv \\
        --binder_chain A --target_chain B \\
        --cpus 8

Usage (from pipeline.py):
    Called automatically when fastrelax_cpus + fastrelax_python are set in YAML.
"""

import argparse
import csv
import glob
import os
import sys


# ---------------------------------------------------------------------------
# Worker (one PDB per spawned process)
# ---------------------------------------------------------------------------

def _relax_one(task):
    """
    Process a single PDB.  Returns a list of maturation candidate rows:
        [{"pdb_name": ..., "chain": ..., "residue": ..., "aa": ..., "reu": ...}, ...]
    Returns an empty list if the pose is skipped or no residues pass the cutoff.
    """
    (pdb_path, output_dir, binder_chain, target_chain,
     score_cutoff, reu_cutoff, script_dir) = task

    # -- BioPython interface detection (no PyRosetta needed yet) ------------
    sys.path.insert(0, script_dir)
    from biopython_utils import hotspot_residues

    interface_res = hotspot_residues(
        pdb_path, target_chain, binder_chain, atom_distance_cutoff=4.0
    )
    if not interface_res:
        print(f"[fastrelax] WARNING: no interface residues in "
              f"{os.path.basename(pdb_path)} – skipping.")
        return []

    # -- PyRosetta (init inside worker) --------------------------------------
    import pyrosetta
    from pyrosetta import rosetta
    from pyrosetta.rosetta.protocols.relax import FastRelax

    pyrosetta.init(
        "-mute all "
        "-ignore_unrecognized_res true "
        "-ignore_zero_occupancy false",
        silent=True,
    )

    name = os.path.splitext(os.path.basename(pdb_path))[0]
    print(f"[fastrelax] Starting  {name}  "
          f"({len(interface_res)} interface residues)")

    try:
        pose = pyrosetta.pose_from_pdb(pdb_path)
    except Exception as e:
        print(f"[fastrelax] ERROR loading {name}: {e}")
        return []

    sfxn = pyrosetta.get_fa_scorefxn()
    pdb_info = pose.pdb_info()

    # -- Convert PDB interface residue numbers → Rosetta pose indices --------
    RS = rosetta.core.select.residue_selector

    pose_indices = []
    for resnum in interface_res:
        idx = pdb_info.pdb2pose(binder_chain, resnum)
        if idx != 0:
            pose_indices.append(idx)

    if not pose_indices:
        print(f"[fastrelax] ERROR: could not map interface residues for {name} – skipping.")
        return []

    # -- Neighborhood selector seeded on binder interface residues -----------
    index_sel = RS.ResidueIndexSelector(",".join(str(i) for i in pose_indices))

    nbrhood_sel = RS.NeighborhoodResidueSelector()
    nbrhood_sel.set_focus_selector(index_sel)
    nbrhood_sel.set_distance(20.0)
    nbrhood_sel.set_include_focus_in_subset(True)

    others_sel = RS.NotResidueSelector(nbrhood_sel)

    # -- Task factory --------------------------------------------------------
    TF = rosetta.core.pack.task
    tf = TF.TaskFactory()
    prevent_rlt = TF.operation.PreventRepackingRLT()
    tf.push_back(TF.operation.OperateOnResidueSubset(prevent_rlt, others_sel))
    tf.push_back(TF.operation.RestrictToRepacking())

    # -- MoveMap -------------------------------------------------------------
    mm = rosetta.core.kinematics.MoveMap()
    mm.set_bb(False)
    mm.set_chi(False)
    for i in range(1, pose.total_residue() + 1):
        chain_id = pdb_info.chain(i)
        if chain_id == binder_chain:
            mm.set_bb(i, True)
            mm.set_chi(i, True)
        else:
            mm.set_bb(i, False)
            mm.set_chi(i, True)

    # -- FastRelax -----------------------------------------------------------
    fr = FastRelax(standard_repeats=2)
    fr.set_scorefxn(sfxn)
    fr.set_task_factory(tf)
    fr.set_movemap(mm)
    fr.min_type("lbfgs_armijo_nonmonotone")
    fr.max_iter(200)
    fr.apply(pose)

    # -- ScoreCutoffFilter ---------------------------------------------------
    total_score = sfxn(pose)
    if total_score > score_cutoff:
        print(f"[fastrelax] SKIP {name}: total_score {total_score:.1f} "
              f"exceeds cutoff {score_cutoff:.0f}")
        return []

    out_path = os.path.join(output_dir, f"{name}_relaxed.pdb")
    pose.dump_pdb(out_path)
    print(f"[fastrelax] Saved   {out_path}  (score={total_score:.2f})")

    # -- Per-residue interface energy analysis (inspired by iam_analyzer) ----
    # Score the pose to populate the energy graph, then walk each interface
    # residue pair and sum edge energies (same logic as energy_interacting_residues
    # in pyrosetta_utils.py).
    sfxn(pose)
    energy_graph = pose.energies().energy_graph()

    maturation_rows = []
    for binder_resnum, data in interface_res.items():
        b_idx = pdb_info.pdb2pose(binder_chain, binder_resnum)
        if b_idx == 0:
            continue

        t_indices = [
            pdb_info.pdb2pose(target_chain, r)
            for r in data["contacts"]
            if pdb_info.pdb2pose(target_chain, r) != 0
        ]
        if not t_indices:
            continue

        dg = 0.0
        for t_idx in t_indices:
            edge = energy_graph.find_edge(b_idx, t_idx)
            if edge is not None:
                dg += edge.dot(sfxn.weights())

        if dg < reu_cutoff:
            maturation_rows.append(binder_resnum)

    print(f"[fastrelax] {name}: {len(maturation_rows)} residues with REU < {reu_cutoff}")
    return (name, os.path.abspath(pdb_path), maturation_rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(input_dir: str, output_dir: str, binder_chain: str, target_chain: str,
        cpus: int, score_cutoff: float, maturation_csv: str,
        reu_cutoff: float = -5.0) -> None:
    """Importable entry point called by pipeline.py."""
    os.makedirs(output_dir, exist_ok=True)

    pdb_files = sorted(glob.glob(os.path.join(input_dir, "*.pdb")))
    if not pdb_files:
        print(f"[fastrelax] No PDB files found in {input_dir} – nothing to do.")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))

    print(
        f"\n[fastrelax] Relaxing {len(pdb_files)} PDBs across {cpus} CPU(s)\n"
        f"  input         : {input_dir}\n"
        f"  output        : {output_dir}\n"
        f"  binder        : chain {binder_chain}\n"
        f"  target        : chain {target_chain}\n"
        f"  maturation csv: {maturation_csv}\n"
    )

    tasks = [
        (pdb, output_dir, binder_chain, target_chain,
         score_cutoff, reu_cutoff, script_dir)
        for pdb in pdb_files
    ]

    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=cpus) as pool:
        results = pool.map(_relax_one, tasks)

    # Build {binder_name: [res1, res2, ...]} dict from worker results
    # Workers return (name, [residues]) or [] if the pose was skipped.
    maturation = {}
    for result in results:
        if not result:
            continue
        name, relaxed_path, residues = result
        maturation[name] = {"path": relaxed_path, "residues": residues}

    def _family(binder_name: str) -> str:
        """nanobody_3_0_relaxed → nanobody_3"""
        base = binder_name.removesuffix("_relaxed")
        return base.rsplit("_", 1)[0]

    # Write maturation.csv — one row per binder, residues as a list
    os.makedirs(os.path.dirname(maturation_csv), exist_ok=True)
    with open(maturation_csv, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["binder_name", "proteinmpnn_family", "path", "maturation_residues"]
        )
        writer.writeheader()
        for binder_name in sorted(maturation):
            writer.writerow({
                "binder_name":         binder_name,
                "proteinmpnn_family":  _family(binder_name),
                "path":                maturation[binder_name]["path"],
                "maturation_residues": maturation[binder_name]["residues"],
            })

    total_res = sum(len(v["residues"]) for v in maturation.values())
    print(f"\n[fastrelax] Done. Relaxed PDBs written to {output_dir}")
    print(f"[fastrelax] Maturation candidates ({total_res} residues across "
          f"{len(maturation)} designs, REU < {reu_cutoff}) → {maturation_csv}")

    # ── Merge families → partial_flow_ready/ ─────────────────────────────────
    # For each proteinmpnn_family: union maturation residues across all members,
    # pick the first member's PDB as the representative structure, copy it to
    # partial_flow_ready/<family>.pdb.  Chain IDs are preserved from the source
    # PDB (not renumbered) to stay consistent with the input YAML.
    import shutil

    partial_flow_dir = os.path.join(os.path.dirname(maturation_csv), "partial_flow_ready")
    os.makedirs(partial_flow_dir, exist_ok=True)

    # Aggregate per family (insertion order gives us sorted first-member by name)
    families: dict = {}
    for binder_name in sorted(maturation):
        fam = _family(binder_name)
        if fam not in families:
            families[fam] = {"path": maturation[binder_name]["path"], "residues": set()}
        families[fam]["residues"].update(maturation[binder_name]["residues"])

    # Drop families where the merged residue list is empty
    dead = [fam for fam, data in families.items() if not data["residues"]]
    for fam in dead:
        print(f"[fastrelax] Family {fam}: no maturation residues – excluded from partial_flow_ready.")
        del families[fam]

    # Write merged_residues.csv
    merged_csv = os.path.join(partial_flow_dir, "merged_residues.csv")
    with open(merged_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["family", "path", "merged_residues"])
        writer.writeheader()
        for fam, data in sorted(families.items()):
            writer.writerow({
                "family":           fam,
                "path":             os.path.join(partial_flow_dir, f"{fam}.pdb"),
                "merged_residues":  sorted(data["residues"]),
            })

    # Copy representative PDB for each family
    for fam, data in families.items():
        src = data["path"]
        if not os.path.isfile(src):
            print(f"[fastrelax] WARNING: {src} not found for family {fam} – skipping.")
            continue
        dst = os.path.join(partial_flow_dir, f"{fam}.pdb")
        shutil.copy2(src, dst)
        print(f"[fastrelax] Family {fam}: residues {sorted(data['residues'])} → {dst}")

    print(f"[fastrelax] Partial-flow ready: {len(families)} PDBs + merged_residues.csv → {partial_flow_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Parallel interface FastRelax (PyRosetta) + maturation residue analysis."
    )
    parser.add_argument("--input_dir",      required=True,  help="Directory of input PDB files.")
    parser.add_argument("--output_dir",     required=True,  help="Directory for relaxed PDB output.")
    parser.add_argument("--maturation_csv", required=True,  help="Path to write maturation.csv.")
    parser.add_argument("--binder_chain",   default="A",    help="Chain ID of the designed binder (default: A).")
    parser.add_argument("--target_chain",   default="B",    help="Chain ID of the target/antigen (default: B).")
    parser.add_argument("--cpus",           type=int, default=4, help="Number of parallel worker processes.")
    parser.add_argument("--score_cutoff",   type=float, default=5_000_000.0,
                        help="Discard poses with total_score above this (default: 5e6).")
    parser.add_argument("--reu_cutoff",     type=float, default=-5.0,
                        help="Interface REU threshold for maturation candidates (default: -5.0).")
    args = parser.parse_args()

    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        binder_chain=args.binder_chain,
        target_chain=args.target_chain,
        cpus=args.cpus,
        score_cutoff=args.score_cutoff,
        maturation_csv=args.maturation_csv,
        reu_cutoff=args.reu_cutoff,
    )


if __name__ == "__main__":
    main()
