# Apptainer Support for SLURM

This directory contains a minimal Apptainer setup for running PPIFlow on GPU-backed SLURM clusters.

## What is included

- `ppiflow.def`: CUDA 12.1 image definition that creates the `ppiflow` environment from the repo's `environment.yml`
- `build_image.sh`: local image build helper
- `run_pipeline.sh`: wrapper for `pipeline.py` inside the container
- `shell.sh`: opens an interactive shell in the image with the repo mounted at `/workspace`
- `examples/submit_*.slurm`: batch job examples
- `examples/pipeline_*.example.yaml`: example pipeline configs that use cluster-safe container paths

## Expected host layout

You should keep the following on shared storage that your SLURM nodes can see:

- this repo
- a `checkpoints/` directory containing downloaded model weights
- your input PDBs
- an `outputs/` directory for generated samples

Recommended structure:

```text
PPIFlow_sjw44/
├── apptainer/
├── checkpoints/
│   ├── binder.ckpt
│   ├── antibody.ckpt
│   ├── nanobody.ckpt
│   └── monomer.ckpt
└── outputs/
```

## Build

Run from the repo root:

```bash
bash apptainer/build_image.sh
```

If your site requires rootless builds, use the site-specific equivalent, for example `apptainer build --fakeroot ...`.

If the build fails with `No space left on device`, point Apptainer temp and cache at a larger scratch filesystem:

```bash
mkdir -p /path/to/scratch/apptainer-tmp /path/to/scratch/apptainer-cache
APPTAINER_TMPDIR=/path/to/scratch/apptainer-tmp \
APPTAINER_CACHEDIR=/path/to/scratch/apptainer-cache \
bash apptainer/build_image.sh /path/to/scratch/ppiflow_cuda121.sif
```

You can also set `APPTAINER_BUILD_ROOT=/path/to/scratch/ppiflow-build` and let the wrapper create `tmp/` and `cache/` under it.

## Interactive run

Edit one of the example YAMLs so its input PDB paths and checkpoint names match your files, then run:

```bash
bash apptainer/run_pipeline.sh \
  apptainer/ppiflow_cuda121.sif \
  apptainer/examples/pipeline_nanobody.example.yaml \
  outputs/nanobody_demo \
  checkpoints \
  5
```

Inside the container the repo is mounted at `/workspace`, outputs at `/output`, and checkpoints at `/checkpoints`.

## SLURM run

Adjust partition, GPU, memory, and walltime in the example job scripts, then submit:

```bash
sbatch apptainer/examples/submit_nanobody.slurm
```

or

```bash
sbatch apptainer/examples/submit_binder.slurm
```

## Notes

- Use `apptainer exec --nv` or `apptainer shell --nv` on GPU nodes.
- The example YAMLs intentionally reference `/workspace/...` and `/checkpoints/...` because those are the runtime bind points used by the wrapper scripts.
- The image does not bake in checkpoints or your mutable inputs.
- `environment.yml` includes several pip packages from external indexes, so image build requires outbound network access.
- The build can need tens of GB of temporary space because it downloads a CUDA base image and a large Python environment.
