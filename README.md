# PPIFlow User Guide

![](./model.png)

PPIFlow is a flow-matching-based framework for de novo generation of high-affinity protein binders targeting precise epitopes. It supports PPI binders, nanobodies, antibodies, motif scaffolding, and unconditional monomer generation. [PPIFlow paper](https://doi.org/10.64898/2026.01.19.700484)

A **demo** of the complete VHH workflow (backbone generation → sequence design → affinity evaluation) is provided in **`demo_vhh.ipynb`**.

[Colab](https://colab.research.google.com/drive/1rWt0E1ZQfKaf2dCFGbqVNau9XWcxbhCd?usp=drive_link) is also available for quickly trying out the model.

---

## Environment Setup

```bash
git clone https://github.com/Mingchenchen/PPIFlow.git
cd PPIFlow
conda env create -f environment.yml
conda activate ppiflow
```

### Download Model Checkpoints
Download from [Google Drive](https://drive.google.com/drive/folders/1BcIBUL2yq1gOchHfN68-AcZK3hiMAMVN?usp=drive_link)

| Task | Checkpoint |
|------|------------|
| Binder | `binder.ckpt` |
| Antibody | `antibody.ckpt` |
| Nanobody | `nanobody.ckpt` |
| Monomer / Motif Scaffolding | `monomer.ckpt` |

---

## Running via pipeline.py (recommended)

`pipeline.py` is the unified entry point. Fill in a YAML config and run:

```bash
python pipeline.py \
    --config configs/pipeline_nanobody.yaml \
    --output /path/to/output \
    --num_samples 5
```

| CLI flag | Description |
|----------|-------------|
| `--config` | Path to a pipeline YAML (see `configs/pipeline_*.yaml`) |
| `--output` | Output directory |
| `--num_samples` | Number of structures to generate (default: 5) |

Template configs for every task live in `configs/`:

| File | Task |
|------|------|
| `pipeline_binder.yaml` | Binder design |
| `pipeline_antibody.yaml` | Antibody CDR design |
| `pipeline_nanobody.yaml` | Nanobody CDR design |
| `pipeline_monomer.yaml` | Monomer generation / motif scaffolding |
| `pipeline_partial_flow_antibody.yaml` | Partial-flow antibody/nanobody refinement |
| `pipeline_partial_flow_binder.yaml` | Partial-flow binder refinement |

Each YAML has a `task:` field that selects the mode. The underlying scripts in `scripts/` are called automatically.

---

## Task YAML Reference

### Binder (`task: binder`)

```yaml
task: binder
input_pdb: /path/to/target.pdb     # or input_csv
target_chain: B
binder_chain: A
specified_hotspots: "B119,B141,B200"
samples_min_length: 75
samples_max_length: 76
config: /path/to/configs/inference_binder.yaml
model_weights: /path/to/binder.ckpt
name: IL7Ra
```

### Antibody (`task: antibody`)

```yaml
task: antibody
antigen_pdb: /path/to/antigen.pdb
framework_pdb: /path/to/antibody_framework.pdb   # CDR loops removed, IMGT numbering
antigen_chain: C
heavy_chain: A
light_chain: B
specified_hotspots: "C11,C14,C15,C101,C107,C108"
cdr_length: "CDRH1,8-8,CDRH2,8-8,CDRH3,10-20,CDRL1,6-9,CDRL2,3-3,CDRL3,9-11"
config: /path/to/configs/inference_nanobody.yaml
model_weights: /path/to/antibody.ckpt
name: 1IJZ_IL13
```

### Nanobody (`task: nanobody`)

Same as antibody but omit `light_chain`:

```yaml
task: nanobody
antigen_pdb: /path/to/antigen.pdb
framework_pdb: /path/to/nanobody_framework.pdb
antigen_chain: C
heavy_chain: A
specified_hotspots: "C101,C135,C171,C198"
cdr_length: "CDRH1,8-8,CDRH2,8-8,CDRH3,9-21"
config: /path/to/configs/inference_nanobody.yaml
model_weights: /path/to/nanobody.ckpt
name: 1CVS_FGFR1
```

### Monomer (`task: monomer`) / Motif Scaffolding (`task: motif_scaffolding`)

Use exactly one of `length_subset` (unconditional) or `motif_csv` (scaffolding):

```yaml
task: monomer
length_subset: "[50, 100]"          # unconditional mode
# motif_csv: /path/to/motifs.csv   # scaffolding mode
# motif_names: "['01_1LDB']"
config: /path/to/configs/inference_unconditional.yaml
model_weights: /path/to/monomer.ckpt
```

### Partial Flow – Antibody/Nanobody (`task: partial_flow_antibody` / `partial_flow_nanobody`)

```yaml
task: partial_flow_antibody
complex_pdb: /path/to/complex.pdb
fixed_positions: "A97-111"
cdr_position: "A26-33,A51-58,A97-111,B149-155,B173-175,B212-221"
antigen_chain: C
heavy_chain: A
light_chain: B          # omit for nanobody
specified_hotspots: "C320"
start_t: 0.5            # 0–1; higher = more structure preserved
retry_Limit: 10
config: /path/to/configs/inference_nanobody.yaml
model_weights: /path/to/antibody.ckpt
name: my_run
```

### Partial Flow – Binder (`task: partial_flow_binder`)

```yaml
task: partial_flow_binder
input_pdb: /path/to/complex.pdb
target_chain: B
binder_chain: A
fixed_positions: "A98-109"
specified_hotspots: "B119,B141"
start_t: 0.7
config: /path/to/configs/inference_binder_partial.yaml
model_weights: /path/to/binder.ckpt
name: my_run
```

---

## Key Format Notes

**Hotspots** – chain prefix on every residue, must match `antigen_chain`/`target_chain`:
```
"C11,C14,C101"
```

**CDR length** – comma-separated name/range pairs, one per CDR:
```
"CDRH1,8-8,CDRH2,8-8,CDRH3,10-20,CDRL1,6-9,CDRL2,3-3,CDRL3,9-11"
```

**`start_t`** – partial flow noise level (0.0–1.0). Higher values refine gently; lower values redesign more aggressively.

---

## Direct Script Usage

The underlying scripts in `scripts/` can still be called directly if preferred:

```bash
python scripts/sample_binder.py \
    --input_pdb /path/to/target.pdb \
    --target_chain B \
    --binder_chain A \
    --config /path/to/configs/inference_binder.yaml \
    --specified_hotspots "B119,B141,B200" \
    --samples_min_length 75 \
    --samples_max_length 76 \
    --samples_per_target 5 \
    --model_weights /path/to/binder.ckpt \
    --output_dir /path/to/output \
    --name IL7Ra
```

---

## Cite

```
@article{yu2026ppiflow,
    author = {Yu, Qilin and Guo, Liangyue and Qin, Xiayan and Huang, Xikun and Tian, Baihui and Wang, Hongzhun and Liu, Yu and Lang, Yunzhi and Wang, Di and Shen, Zhouhanyu and Lin, Jie and Chen, Mingchen},
    title = {High-Affinity Protein Binder Design via Flow Matching and In Silico Maturation},
    year = {2026},
    doi = {10.64898/2026.01.19.700484},
    journal = {bioRxiv}
}
```
