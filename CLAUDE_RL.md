# PPIFlow – Residue Interaction Guidance (RL / Gradient Steering)

## Residue Interaction Guidance (`models/interaction_guidance.py`)

Added by Claude. Steers the flow-matching generator at inference time so a
specific binder residue type (e.g. ARG) is geometrically favoured to sit
opposite a chosen hotspot residue type (e.g. ASP). **No retraining required.**

### How it works

PPIFlow generates backbone frames only (Cα positions + rotation matrices).
Sequence is assigned afterwards by ProteinMPNN. The guidance nudges the
backbone into conformations where ProteinMPNN is more likely to place the
desired residue, by exploiting the fact that ProteinMPNN is sensitive to
local backbone geometry when choosing which amino acid to place.

At each reverse-diffusion step the model emits a "clean" predicted structure
(`pred_trans_1`, `pred_rotmats_1`). Guidance adds a gradient step on top of
that prediction before the Euler update is taken:

```
score  = geometric_interaction_score(pred_trans_1, ...)
grad   = ∂score / ∂pred_trans_1          # autograd; zeroed on target residues
pred_trans_1 += guidance_scale * (grad / ||grad||)
```

This is the same mechanism as **classifier guidance** in DDPM/flow matching —
the model's trajectory is steered uphill on the reward landscape at every step.

### Score / reward function

For each hotspot residue **h** on the target and each binder residue **b**
within `ca_cutoff` Å (Cα–Cα):

```
pair_score(b, h) = dist_term(b, h)  +  orient_weight × orient_term(b, h)

dist_term   = exp( -0.5 × ((d_CbCb - d_opt) / sigma_d)² )
orient_term = ReLU( cos( CA_b→Cb_b ,  Cb_b→Cb_h ) )
```

- **`d_CbCb`** — Cβ–Cβ distance between binder residue and hotspot.
- **`d_opt`** — optimal Cβ–Cβ distance for the (desired_binder_aa, hotspot_aa)
  pair, from a lookup table of salt bridges, cation-π, and H-bond pairs.
- **`orient_term`** — cosine between the binder's CA→Cβ direction and the
  Cβ_binder→Cβ_hotspot approach direction.  Positive only when the sidechain
  points *toward* the hotspot.
- Per hotspot, the **max** over nearby binder residues is taken (reward one
  good contact, not an average over all).

Cβ positions are estimated from backbone frames:
`Cb ≈ R @ [0.5312, -0.7728, -1.2024] + CA`  (ideal L-amino-acid geometry).
GLY (no Cβ) uses CA instead.

### Optimal Cβ–Cβ distance table

| Interaction | Pair examples | d_opt (Å) |
|---|---|---|
| Salt bridge | ARG–ASP, ARG–GLU, LYS–ASP, LYS–GLU | 6.5 |
| Cation–π    | ARG–PHE, ARG–TRP, ARG–TYR, LYS–PHE, LYS–TYR | 5.5 |
| H-bond      | HIS–ASP, HIS–GLU, ASN–ASP, GLN–GLU | 5.0–5.5 |
| H-bond      | SER–ASP, THR–ASP, SER–GLU, THR–GLU | 4.5 |
| Generic     | any other pair | 6.0 |

### Key symbols

| Symbol | Location | Purpose |
|---|---|---|
| `InteractionGuidance` | `models/interaction_guidance.py` | Main class; call `.apply()` in sampling loop |
| `geometric_interaction_score` | same | Differentiable reward; usable stand-alone for post-hoc ranking or REINFORCE |
| `compute_cb_positions` | same | Estimates Cβ from backbone rigid frames |
| `build_guidance_from_args` | same | Builds guidance object from argparse namespace |

### CLI usage (`scripts/sample_binder.py`)

```bash
python scripts/sample_binder.py \
  --input_pdb target.pdb --target_chain A \
  --specified_hotspots A11,A14 \
  --guidance_binder_aa ARG \       # residue to place on binder
  --guidance_hotspot_aa ASP \      # hotspot type (omit = auto from structure)
  --guidance_scale 2.0 \           # strength; tune 0.5–5.0
  --guidance_start_t 0.5 \         # apply only in 2nd half of trajectory
  --guidance_sigma_d 1.5 \         # Gaussian width (Å)
  --guidance_ca_cutoff 12.0 \      # proximity cutoff (Å)
  --model_weights binder.ckpt --output_dir out/
```

### Tuning guide

| Parameter | Effect of increasing | Recommended start |
|---|---|---|
| `guidance_scale` | Stronger steering; may distort fold at very high values | 1.0 |
| `guidance_start_t` | Lower = guidance applied earlier = stronger global effect | 0.5 |
| `guidance_sigma_d` | Wider distance tolerance = softer reward | 1.5 |
| `guidance_ca_cutoff` | Considers more distant binder residues | 12.0 |

### Integration points

- `data/interpolant_binder.py` — `sample()` accepts `guidance=` kwarg; applies
  `.apply()` after each model call, including the final step.
- `models/flow_module_binder.py` — `test_step` passes `self._guidance` (set
  from outside before calling `.test()`) to `interpolant.sample()`.
- The `score()` method of `InteractionGuidance` returns the scalar reward and
  can serve as a **REINFORCE reward** in a future fine-tuning loop.
