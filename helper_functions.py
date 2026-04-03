"""
PPIFlow helper functions
========================
Shared utilities used by pipeline.py.
"""

import json
import os
from argparse import Namespace
from datetime import datetime


# ---------------------------------------------------------------------------
# Pipeline state / resumable logging
# ---------------------------------------------------------------------------

_STATE_FILE = "pipeline_state.json"


class PipelineState:
    """Persist step-completion status to <output_dir>/pipeline_state.json."""

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

    def is_done(self, step: str) -> bool:
        return self._data["steps"].get(step, {}).get("status") == "done"

    def mark_done(self, step: str) -> None:
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
# Namespace builders
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
