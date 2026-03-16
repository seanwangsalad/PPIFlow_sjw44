"""
Residue Interaction Guidance for PPIFlow
=========================================
Steers the flow-matching backbone generator so that a user-specified binder
residue type (e.g. ARG) is geometrically favoured to sit opposite a given
hotspot residue type (e.g. ASP) on the target — without any retraining.

How it works (gradient guidance)
---------------------------------
After the model predicts the clean structure (pred_trans_1, pred_rotmats_1)
at each reverse-diffusion step, we compute a differentiable geometric score
and add its gradient (w.r.t. the binder Cα positions) back to pred_trans_1
before taking the Euler step.  This is the same idea as classifier guidance
in DDPM / flow matching, applied here to residue-level interaction geometry.

The reward / score function can also be used stand-alone (e.g. to rank finished
designs or as the REINFORCE reward in a fine-tuning loop).

Usage example
-------------
    from models.interaction_guidance import InteractionGuidance

    guidance = InteractionGuidance(
        desired_binder_aa="ARG",   # what the binder should present
        desired_hotspot_aa="ASP",  # what the target hotspot is (optional: auto-read from aatype)
        guidance_scale=2.0,        # strength (tune between 0.5 and 5.0)
        start_t=0.5,               # apply only in the second half of the trajectory
        sigma_d=1.5,               # distance tolerance (Å)
        ca_cutoff=12.0,            # max Cα–Cα distance to be considered "nearby" (Å)
    )
    # Inside the sampling loop after model_out = model(batch):
    pred_trans_1 = guidance.apply(
        pred_trans_1, pred_rotmats_1,
        trans_1, rotmats_1,
        diffuse_mask, hotspot_mask,
        aatype, t_current,
    )

Amino-acid index convention (AF2 / OpenFold)
--------------------------------------------
A=0  R=1  N=2  D=3  C=4  Q=5  E=6  G=7  H=8  I=9
L=10 K=11 M=12 F=13 P=14 S=15 T=16 W=17 Y=18 V=19
X/mask = 20 or 21
"""

import torch
import torch.nn.functional as F
from typing import Optional, Union

# ---------------------------------------------------------------------------
# Amino-acid lookup tables
# ---------------------------------------------------------------------------

AA1_TO_IDX: dict[str, int] = {
    "A": 0, "R": 1, "N": 2, "D": 3, "C": 4,
    "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9,
    "L": 10, "K": 11, "M": 12, "F": 13, "P": 14,
    "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19,
}

AA3_TO_IDX: dict[str, int] = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
    "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
}

IDX_TO_AA3: dict[int, str] = {v: k for k, v in AA3_TO_IDX.items()}

GLY_IDX = 7  # glycine — has no Cβ, we fall back to Cα

# Ideal Cβ position in the AF2/OpenFold backbone local frame (CA at origin).
# Derived from ideal L-amino-acid geometry.  GLY uses CA (zero offset).
_CB_LOCAL = torch.tensor([0.5312, -0.7728, -1.2024], dtype=torch.float32)

# ---------------------------------------------------------------------------
# Optimal Cβ–Cβ distances (Å) for known interaction types.
# Values are approximate medians from PDB structural statistics.
# Tolerance (sigma_d) is set at construction time.
# ---------------------------------------------------------------------------
_OPTIMAL_CBCB: dict[tuple[int, int], float] = {
    # Salt bridges
    (1, 3): 6.5,   # ARG–ASP
    (3, 1): 6.5,
    (1, 6): 6.5,   # ARG–GLU
    (6, 1): 6.5,
    (11, 3): 6.5,  # LYS–ASP
    (3, 11): 6.5,
    (11, 6): 6.5,  # LYS–GLU
    (6, 11): 6.5,
    # Cation–π
    (1, 13): 5.5,  # ARG–PHE
    (13, 1): 5.5,
    (1, 17): 5.5,  # ARG–TRP
    (17, 1): 5.5,
    (1, 18): 5.5,  # ARG–TYR
    (18, 1): 5.5,
    (11, 13): 5.5, # LYS–PHE
    (13, 11): 5.5,
    (11, 18): 5.5, # LYS–TYR
    (18, 11): 5.5,
    # Hydrogen bonds (polar–charged)
    (8, 3): 5.0,   # HIS–ASP
    (3, 8): 5.0,
    (8, 6): 5.0,   # HIS–GLU
    (6, 8): 5.0,
    (2, 3): 5.0,   # ASN–ASP
    (3, 2): 5.0,
    (5, 6): 5.5,   # GLN–GLU
    (6, 5): 5.5,
    (15, 3): 4.5,  # SER–ASP
    (3, 15): 4.5,
    (16, 3): 4.5,  # THR–ASP
    (3, 16): 4.5,
    (15, 6): 4.5,  # SER–GLU
    (6, 15): 4.5,
    (16, 6): 4.5,  # THR–GLU
    (6, 16): 4.5,
}
_DEFAULT_CBCB = 6.0  # generic contact fallback


def _parse_aa(aa_str: str) -> int:
    """Accept 1-letter or 3-letter amino-acid code and return AF2 integer index."""
    s = aa_str.strip().upper()
    if s in AA1_TO_IDX:
        return AA1_TO_IDX[s]
    if s in AA3_TO_IDX:
        return AA3_TO_IDX[s]
    raise ValueError(
        f"Unknown amino-acid code {aa_str!r}. "
        f"Use 1-letter (e.g. 'R') or 3-letter (e.g. 'ARG')."
    )


def _optimal_cbcb(binder_idx: int, target_idx: int) -> float:
    return _OPTIMAL_CBCB.get(
        (binder_idx, target_idx),
        _OPTIMAL_CBCB.get((target_idx, binder_idx), _DEFAULT_CBCB),
    )


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------

def compute_cb_positions(
    trans: torch.Tensor,
    rotmats: torch.Tensor,
    aatype: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Estimate Cβ positions from backbone rigid frames.

    Args:
        trans:   [B, N, 3]    CA positions (Å)
        rotmats: [B, N, 3, 3] backbone rotation matrices (AF2 convention)
        aatype:  [B, N]       residue type indices; GLY (7) → use CA instead

    Returns:
        cb: [B, N, 3]  estimated Cβ positions
    """
    cb_local = _CB_LOCAL.to(trans.device)
    # Global Cβ = R @ cb_local + CA
    cb = torch.einsum("bnij,j->bni", rotmats, cb_local) + trans
    if aatype is not None:
        gly_mask = (aatype == GLY_IDX).unsqueeze(-1)  # [B, N, 1]
        cb = torch.where(gly_mask, trans, cb)
    return cb


def geometric_interaction_score(
    pred_trans: torch.Tensor,       # [B, N, 3] – full system (binder+target)
    pred_rotmats: torch.Tensor,     # [B, N, 3, 3]
    trans_1: torch.Tensor,          # [B, N, 3] – fixed target positions
    rotmats_1: torch.Tensor,        # [B, N, 3, 3]
    diffuse_mask: torch.Tensor,     # [B, N] – 1=binder, 0=target
    hotspot_mask: torch.Tensor,     # [B, N] – 1=hotspot on target
    aatype: torch.Tensor,           # [B, N] – known AA types (target); binder = 20/21
    desired_binder_aa_idx: int,
    sigma_d: float = 1.5,
    ca_cutoff: float = 12.0,
    orient_weight: float = 0.3,
) -> torch.Tensor:
    """Differentiable geometric score rewarding binder conformations where at
    least one binder residue is well-positioned for `desired_binder_aa_idx` to
    interact with each hotspot residue.

    Score contributions:
      - Distance term: Gaussian centered on the optimal Cβ–Cβ distance for the
        (desired_binder_aa, hotspot_aa) pair.
      - Orientation term: the binder CA→Cβ vector should point toward the
        hotspot Cβ (favours the sidechain facing the right direction).

    The score is summed over (batch × hotspot) and the max over nearby binder
    residues is taken per hotspot, so the model is rewarded for having *one
    good contact* rather than spreading density across all binders.

    Returns:
        score: scalar tensor (larger = better; gradient points uphill)
    """
    B, N = pred_trans.shape[:2]
    device = pred_trans.device

    # ---- Build mixed-frame positions:
    #   binder residues → use predicted positions (diff w.r.t. pred_trans)
    #   target residues → use fixed ground-truth positions
    dm = diffuse_mask.unsqueeze(-1)          # [B, N, 1]
    dm3 = diffuse_mask.unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]

    mixed_trans = pred_trans * dm + trans_1 * (1 - dm)       # [B, N, 3]
    mixed_rots = pred_rotmats * dm3 + rotmats_1 * (1 - dm3)  # [B, N, 3, 3]

    # Cβ positions for the full system
    cb = compute_cb_positions(mixed_trans, mixed_rots, aatype)  # [B, N, 3]

    # ---- Binder Cβ and CA (only diffuse_mask==1 rows)
    binder_mask = diffuse_mask.bool()   # [B, N]
    target_hot  = (hotspot_mask > 0.5) & (~binder_mask)  # [B, N]

    # ---- Per-hotspot score
    total_score = pred_trans.new_zeros(())

    for b in range(B):
        b_idx = torch.where(binder_mask[b])[0]  # binder residue indices
        h_idx = torch.where(target_hot[b])[0]   # hotspot residue indices

        if b_idx.numel() == 0 or h_idx.numel() == 0:
            continue

        cb_b = cb[b][b_idx]      # [Nb, 3]  binder Cβ
        ca_b = mixed_trans[b][b_idx]  # [Nb, 3] binder CA
        cb_h = cb[b][h_idx]      # [Nh, 3]  hotspot Cβ

        aa_h = aatype[b][h_idx]  # [Nh]     hotspot AA type

        # Pairwise Cα–Cα distance gate (cheap, non-differentiable cutoff)
        ca_h = mixed_trans[b][h_idx]  # [Nh, 3]
        ca_dist = torch.cdist(ca_b.unsqueeze(0), ca_h.unsqueeze(0))[0]  # [Nb, Nh]
        in_range = (ca_dist < ca_cutoff)  # [Nb, Nh]

        if not in_range.any():
            continue

        # Cβ–Cβ distance
        cbcb_dist = torch.cdist(cb_b.unsqueeze(0), cb_h.unsqueeze(0))[0]  # [Nb, Nh]

        # Optimal distances – one per hotspot residue
        d_opt = torch.tensor(
            [_optimal_cbcb(desired_binder_aa_idx, int(aa_h[j].item()))
             for j in range(h_idx.numel())],
            device=device, dtype=pred_trans.dtype,
        )  # [Nh]

        # Distance score: Gaussian on Cβ–Cβ [Nb, Nh]
        dist_score = torch.exp(
            -0.5 * ((cbcb_dist - d_opt.unsqueeze(0)) / sigma_d) ** 2
        )

        # Orientation score: binder sidechain should point toward hotspot Cβ
        # CA→CB direction for each binder residue [Nb, 3]
        sc_dir = F.normalize(cb_b - ca_b, dim=-1)
        # CB_binder → CB_hotspot direction [Nb, Nh, 3]
        approach = F.normalize(
            cb_h.unsqueeze(0) - cb_b.unsqueeze(1), dim=-1
        )
        # cosine similarity [Nb, Nh]
        cos_sim = (sc_dir.unsqueeze(1) * approach).sum(-1)
        orient_score = torch.clamp(cos_sim, min=0.0)

        # Combined per-pair score
        pair_score = (dist_score + orient_weight * orient_score) * in_range.float()

        # For each hotspot take the max over binder residues
        # (reward having one well-placed residue, not averaging over all)
        per_hotspot, _ = pair_score.max(dim=0)  # [Nh]
        total_score = total_score + per_hotspot.sum()

    return total_score


# ---------------------------------------------------------------------------
# Main guidance class
# ---------------------------------------------------------------------------

class InteractionGuidance:
    """Gradient-based interaction guidance for PPIFlow binder sampling.

    Parameters
    ----------
    desired_binder_aa : str
        1- or 3-letter code for the amino acid you want the binder to place
        at the interacting position (e.g. "ARG" or "R").
    desired_hotspot_aa : str, optional
        1- or 3-letter code of the hotspot residue type on the target.
        If None, the optimal Cβ–Cβ distance is looked up from the structure's
        actual aatype tensor at each hotspot position.
    guidance_scale : float
        Strength of the guidance gradient added to pred_trans_1.
        Typical range: 0.5 – 5.0. Start low and increase if needed.
    start_t : float
        Fraction of the trajectory at which to begin applying guidance.
        0.0 = from the very start; 0.5 = second half only (recommended).
        Applying too early (low t) tends to disrupt the global fold.
    sigma_d : float
        Width (Å) of the Gaussian distance potential.  Default 1.5 Å.
    ca_cutoff : float
        Maximum Cα–Cα distance (Å) for a binder residue to be considered
        "nearby" a hotspot.  Default 12.0 Å.
    orient_weight : float
        Relative weight of the sidechain-orientation term vs. the distance
        term.  Default 0.3.
    normalize_grad : bool
        If True, the gradient is L2-normalised before scaling, making
        guidance_scale directly interpretable as Å per step.  Default True.
    """

    def __init__(
        self,
        desired_binder_aa: str,
        desired_hotspot_aa: Optional[str] = None,
        guidance_scale: float = 1.0,
        start_t: float = 0.5,
        sigma_d: float = 1.5,
        ca_cutoff: float = 12.0,
        orient_weight: float = 0.3,
        normalize_grad: bool = True,
    ):
        self.binder_aa_idx = _parse_aa(desired_binder_aa)
        self.hotspot_aa_idx = _parse_aa(desired_hotspot_aa) if desired_hotspot_aa else None
        self.guidance_scale = guidance_scale
        self.start_t = start_t
        self.sigma_d = sigma_d
        self.ca_cutoff = ca_cutoff
        self.orient_weight = orient_weight
        self.normalize_grad = normalize_grad

        binder_name = IDX_TO_AA3.get(self.binder_aa_idx, str(self.binder_aa_idx))
        hotspot_name = (
            IDX_TO_AA3.get(self.hotspot_aa_idx, str(self.hotspot_aa_idx))
            if self.hotspot_aa_idx is not None else "auto"
        )
        print(
            f"[InteractionGuidance] binder={binder_name}, "
            f"hotspot={hotspot_name}, scale={guidance_scale}, "
            f"start_t={start_t}"
        )

    def score(
        self,
        pred_trans: torch.Tensor,
        pred_rotmats: torch.Tensor,
        trans_1: torch.Tensor,
        rotmats_1: torch.Tensor,
        diffuse_mask: torch.Tensor,
        hotspot_mask: torch.Tensor,
        aatype: torch.Tensor,
    ) -> torch.Tensor:
        """Return the scalar interaction score (larger = better).

        Can be used stand-alone to rank finished designs or as a REINFORCE
        reward signal when fine-tuning.
        """
        return geometric_interaction_score(
            pred_trans, pred_rotmats,
            trans_1, rotmats_1,
            diffuse_mask, hotspot_mask,
            aatype,
            desired_binder_aa_idx=self.binder_aa_idx,
            sigma_d=self.sigma_d,
            ca_cutoff=self.ca_cutoff,
            orient_weight=self.orient_weight,
        )

    def apply(
        self,
        pred_trans_1: torch.Tensor,
        pred_rotmats_1: torch.Tensor,
        trans_1: torch.Tensor,
        rotmats_1: torch.Tensor,
        diffuse_mask: torch.Tensor,
        hotspot_mask: torch.Tensor,
        aatype: Optional[torch.Tensor],
        t: float,
    ) -> torch.Tensor:
        """Apply gradient guidance to pred_trans_1.

        If t < start_t the tensor is returned unchanged (guidance is off
        during the very noisy early part of the trajectory).

        Args:
            pred_trans_1:   [B, N, 3] model-predicted clean CA positions
            pred_rotmats_1: [B, N, 3, 3] model-predicted clean rotations
            trans_1:        [B, N, 3] fixed target CA positions (ground truth)
            rotmats_1:      [B, N, 3, 3] fixed target rotations
            diffuse_mask:   [B, N] 1=binder residue (generated), 0=target
            hotspot_mask:   [B, N] 1=hotspot on target
            aatype:         [B, N] residue type indices (target known; binder masked)
            t:              current time step value (float, 0–1)

        Returns:
            pred_trans_1_guided: [B, N, 3] steered CA positions
        """
        if t < self.start_t:
            return pred_trans_1

        if hotspot_mask is None or (hotspot_mask.sum() == 0):
            return pred_trans_1

        if aatype is None:
            # Cannot determine hotspot AA types without aatype; skip guidance
            return pred_trans_1

        # Enable gradient tracking on the binder positions
        pt = pred_trans_1.detach().requires_grad_(True)

        score = geometric_interaction_score(
            pt, pred_rotmats_1.detach(),
            trans_1, rotmats_1,
            diffuse_mask, hotspot_mask,
            aatype,
            desired_binder_aa_idx=self.binder_aa_idx,
            sigma_d=self.sigma_d,
            ca_cutoff=self.ca_cutoff,
            orient_weight=self.orient_weight,
        )

        if score == 0:
            return pred_trans_1

        grad = torch.autograd.grad(score, pt)[0]  # [B, N, 3]

        # Zero out gradient on target residues (diffuse_mask == 0)
        grad = grad * diffuse_mask.unsqueeze(-1)

        if self.normalize_grad:
            grad_norm = grad.norm() + 1e-8
            grad = grad / grad_norm

        guided = pred_trans_1 + self.guidance_scale * grad
        return guided.detach()


# ---------------------------------------------------------------------------
# Convenience: build guidance from a flat config dict / argparse namespace
# ---------------------------------------------------------------------------

def build_guidance_from_args(args) -> Optional[InteractionGuidance]:
    """Build an InteractionGuidance object from CLI args (or None if disabled).

    Expects (optional) attributes on args:
        guidance_binder_aa   : str  e.g. "ARG"
        guidance_hotspot_aa  : str  e.g. "ASP"  (or None)
        guidance_scale       : float
        guidance_start_t     : float
        guidance_sigma_d     : float
        guidance_ca_cutoff   : float
    """
    binder_aa = getattr(args, "guidance_binder_aa", None)
    if not binder_aa:
        return None
    return InteractionGuidance(
        desired_binder_aa=binder_aa,
        desired_hotspot_aa=getattr(args, "guidance_hotspot_aa", None),
        guidance_scale=getattr(args, "guidance_scale", 1.0),
        start_t=getattr(args, "guidance_start_t", 0.5),
        sigma_d=getattr(args, "guidance_sigma_d", 1.5),
        ca_cutoff=getattr(args, "guidance_ca_cutoff", 12.0),
    )
