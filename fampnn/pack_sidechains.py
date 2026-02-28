"""
Minimal sidechain packing with FAMPNN 3.0.

Given a PDB with backbone + sequence (no sidechains), packs sidechains using
the fampnn_0_3 model (FAMPNN 3.0 for sequence design).

Usage:
    python pack_sidechains.py input.pdb output.pdb [checkpoint]
"""
import sys
import torch
from pathlib import Path

from fampnn import sampling_utils
from fampnn.data.data import load_feats_from_pdb, process_single_pdb
from fampnn.model.sd_model import SeqDenoiser


def pack_sidechains(pdb_in: str, pdb_out: str, checkpoint: str = "weights/fampnn_0_3.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = SeqDenoiser(ckpt["model_cfg"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])

    # Load and process PDB
    data = load_feats_from_pdb(pdb_in)
    single = process_single_pdb(data)

    keys = ["x", "aatype", "seq_mask", "missing_atom_mask", "residue_index", "chain_index"]
    batch = {k: single[k].unsqueeze(0).to(device) for k in keys}

    # Sidechain diffusion config (defaults from configs/pack.yaml)
    num_steps = 50
    t_scd = sampling_utils.get_timesteps_from_schedule(
        num_steps=num_steps, mode="linear", t_start=0.0, t_end=1.0
    )
    scd_inputs = {
        "num_steps": num_steps,
        "timesteps": t_scd[None].to(device),
        "step_scale": 1.5,
        "churn_cfg": {"s_churn": 0, "s_noise": 1.0, "s_t_min": 0.01, "s_t_max": 50.0, "num_steps": num_steps},
    }

    # Pack sidechains (sequence is fixed; only sidechain coordinates are predicted)
    with torch.no_grad():
        x_denoised, aatype_denoised, aux = model.sidechain_pack(
            batch["x"],
            batch["aatype"],
            seq_mask=batch["seq_mask"],
            missing_atom_mask=batch["missing_atom_mask"],
            residue_index=batch["residue_index"],
            chain_index=batch["chain_index"],
            scd_inputs=scd_inputs,
        )

    samples = {
        "x_denoised": x_denoised,
        "seq_mask": batch["seq_mask"],
        "missing_atom_mask": torch.zeros_like(batch["missing_atom_mask"]),
        "residue_index": batch["residue_index"],
        "chain_index": batch["chain_index"],
        "pred_aatype": aatype_denoised,
        "psce": aux["psce"],
    }

    Path(pdb_out).parent.mkdir(parents=True, exist_ok=True)
    SeqDenoiser.save_samples_to_pdb(samples, [pdb_out])
    print(f"Saved packed structure to {pdb_out}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python pack_sidechains.py input.pdb output.pdb [checkpoint]")
        sys.exit(1)
    pdb_in = sys.argv[1]
    pdb_out = sys.argv[2]
    checkpoint = sys.argv[3] if len(sys.argv) > 3 else "weights/fampnn_0_3.pt"
    pack_sidechains(pdb_in, pdb_out, checkpoint)
