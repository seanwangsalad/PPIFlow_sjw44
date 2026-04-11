"""
PPIFlow helper functions
========================
Shared utilities used by pipeline.py.
"""

import os
from argparse import Namespace


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def _require(cfg: dict, *keys):
    """Raise a clear error if any required key is missing."""
    for k in keys:
        if cfg.get(k) is None:
            raise ValueError(f"Pipeline config is missing required field: '{k}'")


# ---------------------------------------------------------------------------
# Namespace builders
# ---------------------------------------------------------------------------

def _build_binder_args(cfg: dict, output_dir: str, num_samples: int, resume: bool = False) -> Namespace:
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
        resume=resume,
    )


def _build_antibody_nanobody_args(cfg: dict, output_dir: str, num_samples: int, resume: bool = False) -> Namespace:
    _require(cfg, "antigen_pdb", "antigen_chain", "heavy_chain", "model_weights")
    scaffold_mode = cfg.get("scaffold_pdb") is not None
    if not scaffold_mode:
        _require(cfg, "framework_pdb")
    return Namespace(
        antigen_pdb=cfg["antigen_pdb"],
        # scaffold mode
        scaffold_pdb=cfg.get("scaffold_pdb"),
        scaffold_redesign_residues=cfg.get("scaffold_redesign_residues"),
        scaffold_redesign_lengths=cfg.get("scaffold_redesign_lengths"),
        scaffold_redesign_chains=cfg.get("scaffold_redesign_chains"),
        # legacy mode
        framework_pdb=cfg.get("framework_pdb"),
        cdr_length=cfg.get("cdr_length"),
        antigen_chain=cfg["antigen_chain"],
        heavy_chain=cfg["heavy_chain"],
        light_chain=cfg.get("light_chain"),
        specified_hotspots=cfg.get("specified_hotspots"),
        config=cfg.get("config"),
        samples_per_target=num_samples,
        model_weights=cfg["model_weights"],
        output_dir=output_dir,
        name=cfg.get("name"),
        resume=resume,
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
        retry_Limit=cfg.get("retry_Limit", 10),
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
