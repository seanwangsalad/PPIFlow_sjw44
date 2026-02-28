"""
PPIFlow helper functions
========================
Shared utilities used by pipeline.py.
"""

import glob
import json
import os
from argparse import Namespace
from datetime import datetime

import pandas as pd


# ---------------------------------------------------------------------------
# Pipeline state / resumable logging
# ---------------------------------------------------------------------------

_STATE_FILE = "pipeline_state.json"


class PipelineState:
    """Persist step-completion status to <output_dir>/pipeline_state.json.

    Usage::

        state = PipelineState(output_dir, cfg, resume=False)

        if not state.is_done("binder_gen"):
            run_binder_gen(...)
            state.mark_done("binder_gen")
        else:
            print("skipping binder_gen")

    The JSON file records the task name, when the run started, and a
    ``steps`` dict mapping step names to completion timestamps.
    """

    def __init__(self, output_dir: str, cfg: dict, resume: bool = False):
        self.output_dir = output_dir
        self.path = os.path.join(output_dir, _STATE_FILE)
        os.makedirs(output_dir, exist_ok=True)

        if resume and os.path.exists(self.path):
            with open(self.path) as fh:
                self._data = json.load(fh)
            completed = [s for s, v in self._data.get("steps", {}).items()
                         if v.get("status") == "done"]
            print(f"[pipeline] Resuming run – completed steps: {completed or 'none'}")
        else:
            if resume:
                print(f"[pipeline] --resume set but no state file found at {self.path}; starting fresh.")
            self._data = {
                "task": cfg.get("task", "unknown"),
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "steps": {},
            }
            self._save()

    # ------------------------------------------------------------------

    def is_done(self, step: str) -> bool:
        """Return True if *step* was already completed in a previous run."""
        return self._data["steps"].get(step, {}).get("status") == "done"

    def mark_done(self, step: str) -> None:
        """Record *step* as complete and flush to disk."""
        self._data["steps"][step] = {
            "status": "done",
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._save()
        print(f"[pipeline] ✓ {step}")

    def _save(self) -> None:
        with open(self.path, "w") as fh:
            json.dump(self._data, fh, indent=2)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def _require(cfg: dict, *keys):
    """Raise a clear error if any required key is missing."""
    for k in keys:
        if cfg.get(k) is None:
            raise ValueError(f"Pipeline config is missing required field: '{k}'")


# ---------------------------------------------------------------------------
# Namespace builder – binder task
# ---------------------------------------------------------------------------

def _build_binder_args(cfg: dict, output_dir: str, num_samples: int) -> Namespace:
    _require(cfg, "model_weights")
    if cfg.get("input_pdb") is None and cfg.get("input_csv") is None:
        raise ValueError("Binder task requires either 'input_pdb' or 'input_csv'.")
    return Namespace(
        input_pdb=cfg.get("input_pdb"),
        input_csv=cfg.get("input_csv"),
        target_chain=cfg.get("target_chain"),
        binder_chain=cfg.get("binder_chain"),
        config=cfg.get("config"),
        specified_hotspots=cfg.get("specified_hotspots"),
        sample_hotspot_rate_min=cfg.get("sample_hotspot_rate_min"),
        sample_hotspot_rate_max=cfg.get("sample_hotspot_rate_max"),
        samples_min_length=cfg.get("samples_min_length"),
        samples_max_length=cfg.get("samples_max_length"),
        samples_per_target=num_samples,
        model_weights=cfg["model_weights"],
        output_dir=output_dir,
        name=cfg.get("name"),
    )


# ---------------------------------------------------------------------------
# Namespace builders – remaining task types
# ---------------------------------------------------------------------------

def _build_antibody_nanobody_args(cfg: dict, output_dir: str, num_samples: int) -> Namespace:
    _require(cfg, "antigen_pdb", "framework_pdb", "antigen_chain", "heavy_chain", "model_weights")
    return Namespace(
        antigen_pdb=cfg["antigen_pdb"],
        framework_pdb=cfg["framework_pdb"],
        antigen_chain=cfg["antigen_chain"],
        heavy_chain=cfg["heavy_chain"],
        light_chain=cfg.get("light_chain"),
        specified_hotspots=cfg.get("specified_hotspots"),
        cdr_length=cfg.get("cdr_length"),
        config=cfg.get("config"),
        samples_per_target=num_samples,
        model_weights=cfg["model_weights"],
        output_dir=output_dir,
        name=cfg.get("name"),
    )


def _build_monomer_args(cfg: dict, output_dir: str, num_samples: int) -> Namespace:
    _require(cfg, "model_weights")
    return Namespace(
        config=cfg.get("config"),
        model_weights=cfg["model_weights"],
        output_dir=output_dir,
        length_subset=cfg.get("length_subset"),
        motif_csv=cfg.get("motif_csv"),
        motif_names=cfg.get("motif_names"),
        samples_num=num_samples,
    )


def _build_partial_antibody_nanobody_args(cfg: dict, output_dir: str, num_samples: int) -> Namespace:
    _require(cfg, "complex_pdb", "fixed_positions", "antigen_chain", "heavy_chain", "start_t", "model_weights")
    return Namespace(
        complex_pdb=cfg["complex_pdb"],
        fixed_positions=cfg["fixed_positions"],
        cdr_position=cfg.get("cdr_position"),
        antigen_chain=cfg["antigen_chain"],
        heavy_chain=cfg["heavy_chain"],
        light_chain=cfg.get("light_chain"),
        specified_hotspots=cfg.get("specified_hotspots"),
        start_t=cfg["start_t"],
        samples_per_target=num_samples,
        output_dir=output_dir,
        retry_Limit=cfg.get("retry_Limit"),
        config=cfg.get("config"),
        model_weights=cfg["model_weights"],
        name=cfg.get("name"),
    )


def _build_partial_binder_args(cfg: dict, output_dir: str, num_samples: int) -> Namespace:
    _require(cfg, "model_weights")
    if cfg.get("input_pdb") is None and cfg.get("input_csv") is None:
        raise ValueError("partial_flow_binder requires either 'input_pdb' or 'input_csv'.")
    return Namespace(
        input_pdb=cfg.get("input_pdb"),
        input_csv=cfg.get("input_csv"),
        target_chain=cfg.get("target_chain"),
        binder_chain=cfg.get("binder_chain"),
        config=cfg.get("config"),
        specified_hotspots=cfg.get("specified_hotspots"),
        sample_hotspot_rate_min=cfg.get("sample_hotspot_rate_min"),
        sample_hotspot_rate_max=cfg.get("sample_hotspot_rate_max"),
        interface_dist=cfg.get("interface_dist"),
        fixed_positions=cfg.get("fixed_positions"),
        samples_per_target=num_samples,
        model_weights=cfg["model_weights"],
        start_t=cfg.get("start_t"),
        output_dir=output_dir,
        name=cfg.get("name"),
    )


# ---------------------------------------------------------------------------
# Inverse folding – fixed-positions CSV
# ---------------------------------------------------------------------------

def create_mpnn_fixed_positions_csv(output_dir: str) -> str:
    """Create mpnn_fixed_positions.csv inside output_dir.

    Scans every *.pdb in output_dir and determines the fixed (non-designed)
    residue positions for ProteinMPNN.  Fixed residues are all residue
    positions on the designed chain(s) that have B-factor != 2.0.

    B-factor encoding written by PPIFlow:
        4.0  – framework residues (fixed)
        2.0  – CDR / designed residues
        0.0  – antigen / non-hotspot residues
        1.0  – antigen hotspot residues

    The output CSV is tab-separated with columns:
        pdb_name  <TAB>  motif_index

    ``motif_index`` lists space-separated 1-based residue indices that must
    NOT be redesigned, with a ``-`` separating entries for different designed
    chains (e.g. heavy chain then light chain).  A trailing ``-`` is appended
    for single-designed-chain molecules (nanobody) to keep the format
    consistent.

    Returns:
        Absolute path to the written CSV file.
    """
    pdb_files = sorted(glob.glob(os.path.join(output_dir, "*.pdb")))

    rows = []
    for pdb_file in pdb_files:
        pdb_name = os.path.splitext(os.path.basename(pdb_file))[0]

        # --- parse per-chain residue B-factors ---
        chain_order: list[str] = []
        chain_residues: dict[str, dict[int, float]] = {}  # chain -> {resnum: max_bf}

        with open(pdb_file) as fh:
            for line in fh:
                if not line.startswith("ATOM"):
                    continue
                chain = line[21]
                resnum = int(line[22:26])
                bf = float(line[60:66])
                if chain not in chain_residues:
                    chain_residues[chain] = {}
                    chain_order.append(chain)
                prev = chain_residues[chain].get(resnum, -1.0)
                chain_residues[chain][resnum] = max(prev, bf)

        # --- identify designed chains ---
        # A designed chain (antibody/nanobody heavy or light) always has BOTH:
        #   B-factor 4.0  – framework residues (fixed)
        #   B-factor 2.0  – CDR residues (designable)
        # The target/antigen chain in binder outputs can also reach B-factor 2.0
        # when hotspot_mask + target_interface_mask both equal 1, so requiring
        # both 4.0 and 2.0 avoids falsely treating it as a designed chain.
        def _has_bf(chain, target):
            return any(abs(bf - target) < 0.01 for bf in chain_residues[chain].values())

        designed_chains = [
            c for c in chain_order
            if _has_bf(c, 4.0) and _has_bf(c, 2.0)
        ]

        if not designed_chains:
            # Binder task: entire binder chain is de novo designed, no fixed positions
            rows.append({"pdb_name": pdb_name, "motif_index": ""})
            continue

        # --- for each designed chain, collect fixed residues (B-factor != 2.0) ---
        chain_fixed_parts: list[str] = []
        for chain in designed_chains:
            fixed_resnums = sorted(
                r for r, bf in chain_residues[chain].items()
                if abs(bf - 2.0) >= 0.01
            )
            chain_fixed_parts.append(" ".join(str(r) for r in fixed_resnums))

        # Join chains with '-'; always append trailing '-' for consistency
        motif_index = "-".join(chain_fixed_parts) + "-"

        rows.append({"pdb_name": pdb_name, "motif_index": motif_index})

    out_path = os.path.join(output_dir, "mpnn_fixed_positions.csv")
    df = pd.DataFrame(rows, columns=["pdb_name", "motif_index"])
    df.to_csv(out_path, sep="\t", index=False)
    print(f"[inverse_folding] mpnn_fixed_positions.csv → {out_path}  ({len(rows)} entries)")
    return out_path


def _detect_designed_chains(output_dir: str) -> list[str]:
    """Return the designed chain IDs from the first PDB in output_dir.

    A designed chain (antibody/nanobody heavy or light) has BOTH:
      B-factor 4.0 – framework residues (fixed)
      B-factor 2.0 – CDR residues (designable)

    Returns an empty list for binder outputs, where no chain carries both marks.
    """
    pdb_files = sorted(glob.glob(os.path.join(output_dir, "*.pdb")))
    if not pdb_files:
        return []

    chain_bfs: dict[str, set] = {}
    with open(pdb_files[0]) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            bf = float(line[60:66])
            chain_bfs.setdefault(chain, set()).add(bf)

    return [
        c for c in chain_bfs
        if any(abs(bf - 4.0) < 0.01 for bf in chain_bfs[c])
        and any(abs(bf - 2.0) < 0.01 for bf in chain_bfs[c])
    ]


def run_protein_mpnn(output_dir: str, csv_path: str, chain_list: str, cfg: dict) -> None:
    """Run ProteinMPNN inverse folding on all PDBs in output_dir.

    Sequences are written to <output_dir>/mpnn_output/.

    Args:
        output_dir:  Folder containing the generated backbone PDB files.
        csv_path:    Path to mpnn_fixed_positions.csv produced by
                     create_mpnn_fixed_positions_csv().
        chain_list:  Space-separated chain IDs to design, e.g. ``"A"`` or
                     ``"A B"``.
        cfg:         Pipeline config dict; must contain ``mpnn_weights`` and
                     may contain ``model_name``, ``num_seqs_per_target``,
                     ``batch_size``, ``sampling_temp``, ``mpnn_omit_AAs``.
    """
    import sys
    from argparse import Namespace as _Namespace

    # Put the ProteinMPNN directory on sys.path so its relative imports resolve
    mpnn_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ProteinMPNN")
    if mpnn_dir not in sys.path:
        sys.path.insert(0, mpnn_dir)

    import protein_mpnn_run  # noqa: E402 – lazy import after sys.path setup

    out_folder = os.path.join(output_dir, "mpnn_output")
    os.makedirs(out_folder, exist_ok=True)

    args = _Namespace(
        # ── paths ──────────────────────────────────────────────────────────────
        folder_with_pdbs_path=output_dir,
        out_folder=out_folder,
        path_to_model_weights=cfg["mpnn_weights"],
        # ── design spec ────────────────────────────────────────────────────────
        chain_list=chain_list,
        position_list=csv_path,
        model_name=cfg.get("model_name", "v_48_020"),
        # ── sampling ───────────────────────────────────────────────────────────
        num_seq_per_target=int(cfg.get("num_seqs_per_target", 8)),
        batch_size=int(cfg.get("batch_size", 1)),
        sampling_temp=str(cfg.get("sampling_temp", "0.1")),
        omit_AAs=cfg.get("mpnn_omit_AAs", "X"),
        # ── unused / defaults ──────────────────────────────────────────────────
        suppress_print=0,
        ca_only=False,
        use_soluble_model=False,
        seed=0,
        backbone_noise=0.00,
        max_length=200000,
        save_score=0,
        save_probs=0,
        score_only=0,
        path_to_fasta="",
        conditional_probs_only=0,
        conditional_probs_only_backbone=0,
        unconditional_probs_only=0,
        pdb_path="",
        pdb_path_chains="",
        jsonl_path="",
        chain_id_jsonl="",
        fixed_positions_jsonl="",
        bias_AA_jsonl="",
        bias_by_res_jsonl="",
        omit_AA_jsonl="",
        pssm_jsonl="",
        pssm_multi=0.0,
        pssm_threshold=0.0,
        pssm_log_odds_flag=0,
        pssm_bias_flag=0,
        tied_positions_jsonl="",
    )

    print(f"[inverse_folding] Running ProteinMPNN → {out_folder}")
    protein_mpnn_run.main(args)
