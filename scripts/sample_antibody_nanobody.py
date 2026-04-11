"""
Backbone Structure Sampling Pipeline - Binder Design
Supports both Antibody (heavy + light chain) and Nanobody (heavy chain only)
"""

import os
import re
import random
import dataclasses
import yaml
import time
import numpy as np
import pandas as pd
from Bio import PDB
from typing import Dict, Any, List, Optional
import argparse
from omegaconf import OmegaConf


from experiments.inference_antibody import Experiment
from data import utils as du
from data import parsers
from data import errors


# =============================================================================
# Config Manager
# =============================================================================


class ConfigManager:
    """Manages reading, modifying, and saving YAML configuration files."""

    def __init__(self, default_config_path: str):
        self.default_config = self._load_config(default_config_path)
        self.current_config: Optional[Dict[str, Any]] = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def update_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep update configuration dictionary.
        Example: updates = {'model': {'dropout': 0.2}, 'input': {'pdb_path': 'new_path.pdb'}}
        """

        def deep_update(original: Dict, update: Dict) -> Dict:
            for key, value in update.items():
                if isinstance(value, dict) and key in original:
                    deep_update(original[key], value)
                else:
                    original[key] = value
            return original

        self.current_config = deep_update(
            self.default_config.copy(), updates
        )
        return self.current_config

    def save_config(self, save_path: str) -> None:
        """Save configuration to YAML file."""
        config_to_save = self.current_config
        with open(save_path, "w") as f:
            yaml.dump(
                config_to_save,
                f,
                sort_keys=False,
                default_flow_style=False,
            )
        print(f"Config saved to: {save_path}")


# =============================================================================
# Preprocessing Utilities
# =============================================================================


def extract_features(structure, struct_chains):
    """Extract features from protein structure chains."""
    struct_feats = []
    all_seqs = set()
    for chain_id, chain in struct_chains.items():
        chain_id = du.chain_str_to_int(chain_id)
        chain_prot = parsers.process_chain(chain, chain_id)
        chain_dict = dataclasses.asdict(chain_prot)
        chain_dict = du.parse_chain_feats(
            chain_dict, normalize_positions=False
        )
        all_seqs.add(tuple(chain_dict["aatype"]))
        struct_feats.append(chain_dict)
    return struct_feats


def random_coord() -> List[float]:
    """Generate a random coordinate."""
    return [random.uniform(-1, 1) for _ in range(3)]


def make_virtual_ala(res_id):
    """Construct an ALA residue with full backbone and CB atom."""
    res = PDB.Residue.Residue((" ", res_id, " "), "ALA", "")

    atom_names = ["N", "CA", "C", "O", "CB"]
    for i, atom_name in enumerate(atom_names):
        coord = random_coord()
        atom = PDB.Atom.Atom(
            atom_name,
            coord,
            1.0,
            1.0,
            " ",
            atom_name,
            i,
            element=atom_name[0],
        )
        res.add(atom)
    return res


def count_residues(pdb_file: str) -> int:
    """Count the number of residues in a PDB file."""
    parser = PDB.PDBParser(QUIET=True)
    struct = parser.get_structure("s", pdb_file)[0]
    return sum(1 for _ in struct.get_residues())


# =============================================================================
# PDB Processing Functions
# =============================================================================


def _merge_framework_cdr(
    framework_file_path: str,
    heavy_chain: str,
    cdr_cfg: str,
    antibody_file_path: str,
    light_chain: Optional[str] = None,
) -> List[int]:
    """
    Insert virtual CDR regions into framework and construct antibody_mask.
    Supports antibody (heavy + light) and nanobody (heavy only) modes.
    """
    # Parse CDR length configuration
    parts = cdr_cfg.split(",")
    cdr_length_dict = {}
    for i in range(0, len(parts), 2):
        name = parts[i].strip()
        low, high = map(int, parts[i + 1].split("-"))
        cdr_length_dict[name] = random.randint(low, high)
    print(f"cdr_length_dict: {cdr_length_dict}")

    # Parse framework structure
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("fw", framework_file_path)
    model = structure[0]

    new_model = PDB.Model.Model(0)
    antibody_mask = []

    for chain in model:
        residues = list(chain.get_residues())
        res_ids = [res.get_id()[1] for res in residues]

        # Find gaps between consecutive residues
        gaps = []
        for i in range(len(res_ids) - 1):
            if res_ids[i + 1] != res_ids[i] + 1:
                gaps.append((i, res_ids[i], res_ids[i + 1]))

        # Determine CDR names based on chain type
        if chain.id.upper() == heavy_chain.upper():
            cdr_names = ["CDRH1", "CDRH2", "CDRH3"]
        elif (
            light_chain is not None
            and chain.id.upper() == light_chain.upper()
        ):
            cdr_names = ["CDRL1", "CDRL2", "CDRL3"]
        else:
            raise ValueError(f"Unknown chain: {chain.id}")

        new_chain = PDB.Chain.Chain(chain.id)
        res_counter = 1
        cdr_counter = 0

        for i, res in enumerate(residues):
            # Copy framework residue
            new_res = PDB.Residue.Residue(
                (" ", res_counter, " "), res.resname, res.segid
            )
            for atom in res:
                new_res.add(atom.copy())
            new_chain.add(new_res)
            antibody_mask.append(0)
            res_counter += 1

            # Insert CDR region
            for gap_idx, start, end in gaps:
                if res_ids[i] == start and cdr_counter < len(cdr_names):
                    cdr_len = cdr_length_dict[cdr_names[cdr_counter]]
                    for j in range(cdr_len):
                        ala_res = make_virtual_ala(res_counter)
                        new_chain.add(ala_res)
                        antibody_mask.append(1)
                        res_counter += 1
                    cdr_counter += 1

        new_model.add(new_chain)

    # Save the merged structure
    new_structure = PDB.Structure.Structure("antibody_with_cdr")
    new_structure.add(new_model)
    io = PDB.PDBIO()
    io.set_structure(new_structure)
    io.save(antibody_file_path)

    return antibody_mask


def _merge_pdb_files(
    antigen_pdb: str,
    antibody_pdb: str,
    output_pdb: str,
    antigen_chain: str,
) -> None:
    """Merge antigen and antibody PDB files into a single file."""
    with open(antigen_pdb, "r") as f:
        antigen_lines = f.readlines()
    with open(antibody_pdb, "r") as f:
        antibody_lines = f.readlines()

    merged_lines = []

    # Keep ATOM/HETATM lines from antibody
    for line in antibody_lines:
        if line.startswith(("ATOM", "HETATM")):
            merged_lines.append(line)

    # Keep ATOM/HETATM lines from antigen
    for line in antigen_lines:
        if line.startswith(("ATOM", "HETATM")):
            chain_id = line[21]
            if chain_id != antigen_chain:
                continue
            merged_lines.append(line)

    # Add END marker
    merged_lines.append("END\n")

    with open(output_pdb, "w") as f:
        f.writelines(merged_lines)


def get_indices_from_spec(structure, spec_str: str) -> List[int]:
    """Parse residue indices from specification string."""
    if not spec_str:
        return []
    # Format: "H100,L50,A25" (ChainID + ResidueNumber [+ InsertionCode optional])
    target_ids = set(spec_str.strip().split(","))
    indices = []
    res_idx = 0
    for chain in structure.get_chains():
        cid = chain.id
        for res in chain:
            if PDB.is_aa(res, standard=True):
                res_num = str(res.get_id()[1])
                icode = res.get_id()[2].strip()
                full_id = f"{cid}{res_num}{icode}"
                if full_id in target_ids:
                    indices.append(res_idx)
                res_idx += 1
    return indices


# =============================================================================
# Main Processing Functions
# =============================================================================


def process_file(
    input_info: Dict[str, Any], write_dir: str, sample_id: int
) -> Dict[str, Any]:
    """Process protein file into usable pickles."""
    antigen_file_path = input_info["antigen_pdb"]

    # Generate output file paths
    antibody_file_path = os.path.join(
        write_dir,
        f"{input_info['pdb_name']}_antibody_sample_{sample_id}.pdb",
    )

    # Build scaffold with redesign regions
    if "scaffold_pdb" in input_info:
        antibody_mask = _build_redesigned_scaffold(
            scaffold_pdb=input_info["scaffold_pdb"],
            redesign_residues=input_info["redesign_residues"],
            redesign_lengths=input_info["redesign_lengths"],
            heavy_chain=input_info["heavy_chain"],
            antibody_file_path=antibody_file_path,
            light_chain=input_info.get("light_chain"),
            redesign_chains=input_info.get("redesign_chains"),
        )
    else:
        antibody_mask = _merge_framework_cdr(
            input_info["framework_pdb"],
            input_info["heavy_chain"],
            input_info["cdr_length"],
            antibody_file_path,
            light_chain=input_info.get("light_chain"),
        )

    # Merge antigen and antibody
    pdb_name = input_info["pdb_name"]
    merged_filepath = os.path.join(
        write_dir, f"{pdb_name}_antigen_antibody_sample_{sample_id}.pdb"
    )
    _merge_pdb_files(
        antigen_file_path,
        antibody_file_path,
        merged_filepath,
        input_info["antigen_chain"],
    )

    # Calculate CDR mask
    antibody_len = count_residues(antibody_file_path)
    total_len = count_residues(merged_filepath)
    antigen_len = total_len - antibody_len

    if antigen_len == 0:
        # List actual chains in the antigen PDB for debugging
        _parser = PDB.PDBParser(QUIET=True)
        _ag_struct = _parser.get_structure("ag", antigen_file_path)
        _ag_chains = [c.id for c in _ag_struct[0].get_chains()]
        raise ValueError(
            f"No antigen residues merged! antigen_chain='{input_info['antigen_chain']}' "
            f"not found in {antigen_file_path} (chains present: {_ag_chains}). "
            f"Check that antigen_chain in your YAML matches the chain ID in the antigen PDB."
        )

    cdr_mask = np.array(antibody_mask + [0] * antigen_len, dtype=np.int8)

    # Extract features from merged structure
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(
        f"{pdb_name}_sample_{sample_id}", merged_filepath
    )
    struct_chains = {
        chain.id.upper(): chain for chain in structure.get_chains()
    }
    struct_feats = extract_features(structure, struct_chains)
    complex_feats = du.concat_np_features(struct_feats, False)

    # Parse hotspot indices
    hotspot_index = get_indices_from_spec(
        structure, input_info["hotspots"]
    )

    # Determine binder_id based on chain configuration
    binder_id = input_info["heavy_chain"]
    if input_info.get("light_chain"):
        binder_id = input_info["heavy_chain"] + input_info["light_chain"]

    # Build metadata
    processed_path = os.path.join(
        write_dir, f"{pdb_name}_sample_{sample_id}.pkl"
    )
    metadata = {
        "id": f"{pdb_name}_sample_{sample_id}",
        "pdb_name": f"{pdb_name}_{sample_id}",
        "target_id": input_info["antigen_chain"],
        "binder_id": binder_id,
        "processed_path": processed_path,
        "antigen_path": antigen_file_path,
        "data_level": 1,
        "hotspots": hotspot_index,
    }

    metadata["num_chains"] = len(struct_chains)
    print("Chains in antigen and framework pdb: ", metadata["num_chains"])

    # Process geometry features
    complex_aatype = complex_feats["aatype"]
    metadata["seq_len"] = len(complex_aatype)
    modeled_idx = np.where(complex_aatype != 20)[0]
    if np.sum(complex_aatype != 20) == 0:
        raise errors.LengthError("No modeled residues")
    metadata["modeled_seq_len"] = len(modeled_idx)
    complex_feats["modeled_idx"] = modeled_idx

    # Set chain groups for antigen and binder
    antigen_chains = [
        du.chain_str_to_int(chain_id)
        for chain_id in input_info["antigen_chain"]
    ]
    binder_chains = [du.chain_str_to_int(input_info["heavy_chain"])]
    if input_info.get("light_chain"):
        binder_chains.extend(
            [du.chain_str_to_int(input_info["light_chain"])]
        )

    complex_feats["chain_groups"] = np.where(
        np.isin(complex_feats["chain_index"], antigen_chains),
        0,
        np.where(
            np.isin(complex_feats["chain_index"], binder_chains),
            1,
            complex_feats["chain_index"],
        ),
    )
    complex_feats["cdr_mask"] = cdr_mask
    print(f"complex_feats: {complex_feats.keys()}")
    du.write_pkl(processed_path, complex_feats)

    if os.path.exists(antibody_file_path):
        os.remove(antibody_file_path)
    if os.path.exists(merged_filepath):
        os.remove(merged_filepath)

    return metadata


def _build_redesigned_scaffold(
    scaffold_pdb: str,
    redesign_residues: List[List[int]],
    redesign_lengths: List[List[int]],
    heavy_chain: str,
    antibody_file_path: str,
    light_chain: Optional[str] = None,
    redesign_chains: Optional[List[str]] = None,
) -> List[int]:
    """Build a scaffold with redesign regions replaced by virtual ALA residues.

    Takes a complete scaffold PDB, removes the specified residue ranges, and
    inserts virtual ALA residues of sampled lengths in their place.  Works for
    any number of redesign regions — not limited to 3 CDRs.

    Args:
        scaffold_pdb: Complete scaffold PDB (all residues present).
        redesign_residues: Inclusive [start, end] ranges to redesign,
            e.g. [[26,33],[51,57],[96,110]].
        redesign_lengths: [min, max] length per region, e.g. [[8,8],[8,8],[9,21]].
        heavy_chain: Chain ID of the heavy chain.
        antibody_file_path: Where to write the output PDB.
        light_chain: Chain ID of the light chain (optional, for antibody).
        redesign_chains: Which chain each region belongs to, e.g. ["A","A","A","B","B","B"].
            If omitted, all regions are assigned to heavy_chain.

    Returns:
        antibody_mask: 0 for framework residues, 1 for designed (inserted) residues.
    """
    # Sample a concrete length for each redesign region.
    sampled_lengths = [random.randint(lo, hi) for lo, hi in redesign_lengths]

    parser_obj = PDB.PDBParser(QUIET=True)
    structure = parser_obj.get_structure("scaffold", scaffold_pdb)
    scaffold_chains = [chain.id for chain in structure[0].get_chains()]

    # redesign_chains specifies which SCAFFOLD chain each region belongs to.
    # If omitted: nanobody (single scaffold chain) → all regions on that chain.
    #             antibody (two scaffold chains) → must be explicit.
    if redesign_chains is None:
        if light_chain is None:
            # Nanobody: single scaffold chain, all regions on it.
            redesign_chains = [scaffold_chains[0].upper()] * len(redesign_residues)
        else:
            raise ValueError(
                "scaffold_redesign_chains must be specified for antibody mode. "
                "Provide a list of scaffold chain IDs, one per redesign region, "
                "e.g. scaffold_redesign_chains: [A, A, A, B, B, B]"
            )

    # Build scaffold chain → output chain ID mapping.
    # The output always uses the user-specified heavy_chain / light_chain IDs.
    chain_output_map: dict = {}
    if light_chain is None:
        # Nanobody: map every scaffold chain to heavy_chain (only one chain expected).
        for sc in scaffold_chains:
            chain_output_map[sc.upper()] = heavy_chain.upper()
    else:
        # Antibody: first scaffold chain → heavy_chain, second → light_chain.
        if len(scaffold_chains) >= 1:
            chain_output_map[scaffold_chains[0].upper()] = heavy_chain.upper()
        if len(scaffold_chains) >= 2:
            chain_output_map[scaffold_chains[1].upper()] = light_chain.upper()

    # Map (scaffold_chain_id_upper, res_num) → region index for quick lookup.
    region_of: dict = {}
    for idx, (start, end) in enumerate(redesign_residues):
        chain_id = redesign_chains[idx].upper()
        for res_num in range(start, end + 1):
            region_of[(chain_id, res_num)] = idx

    inserted_regions: set = set()

    new_model = PDB.Model.Model(0)
    antibody_mask: List[int] = []

    for chain in structure[0]:
        cid = chain.id.upper()
        # Use the remapped output chain ID.
        out_chain_id = chain_output_map.get(cid, chain.id)
        new_chain = PDB.Chain.Chain(out_chain_id)
        res_counter = 1

        for res in chain.get_residues():
            key = (cid, res.get_id()[1])

            if key in region_of:
                ridx = region_of[key]
                if ridx not in inserted_regions:
                    # First residue of this region → insert virtual ALAs.
                    for _ in range(sampled_lengths[ridx]):
                        new_chain.add(make_virtual_ala(res_counter))
                        antibody_mask.append(1)
                        res_counter += 1
                    inserted_regions.add(ridx)
                # Skip the original residue (it is being redesigned).
            else:
                # Framework residue — keep it.
                new_res = PDB.Residue.Residue(
                    (" ", res_counter, " "), res.resname, res.segid
                )
                for atom in res:
                    new_res.add(atom.copy())
                new_chain.add(new_res)
                antibody_mask.append(0)
                res_counter += 1

        new_model.add(new_chain)

    new_structure = PDB.Structure.Structure("redesigned_scaffold")
    new_structure.add(new_model)
    io = PDB.PDBIO()
    io.set_structure(new_structure)
    io.save(antibody_file_path)

    print(f"Redesign regions: {len(redesign_residues)}, sampled lengths: {sampled_lengths}")
    return antibody_mask


def _generate_framework_reference(
    scaffold_pdb: str,
    redesign_residues: List[List[int]],
    heavy_chain: str,
    output_path: str,
    redesign_chains: Optional[List[str]] = None,
    light_chain: Optional[str] = None,
) -> None:
    """Write a framework PDB with redesign residues stripped out.

    Used as the RMSD reference in flow_module_antibody.py so the model can
    verify that framework residues didn't drift during generation.
    """
    parser_obj = PDB.PDBParser(QUIET=True)
    structure = parser_obj.get_structure("s", scaffold_pdb)
    scaffold_chains = [chain.id for chain in structure[0].get_chains()]

    if redesign_chains is None:
        redesign_chains = [scaffold_chains[0].upper()] * len(redesign_residues)

    # Same chain remapping as _build_redesigned_scaffold.
    chain_output_map: dict = {}
    if light_chain is None:
        for sc in scaffold_chains:
            chain_output_map[sc.upper()] = heavy_chain.upper()
    else:
        if len(scaffold_chains) >= 1:
            chain_output_map[scaffold_chains[0].upper()] = heavy_chain.upper()
        if len(scaffold_chains) >= 2:
            chain_output_map[scaffold_chains[1].upper()] = light_chain.upper()

    remove: set = set()
    for idx, (start, end) in enumerate(redesign_residues):
        chain_id = redesign_chains[idx].upper()
        for res_num in range(start, end + 1):
            remove.add((chain_id, res_num))

    new_model = PDB.Model.Model(0)
    for chain in structure[0]:
        cid = chain.id.upper()
        out_chain_id = chain_output_map.get(cid, chain.id)
        new_chain = PDB.Chain.Chain(out_chain_id)
        for res in chain.get_residues():
            if (cid, res.get_id()[1]) not in remove:
                new_chain.add(res.copy())
        new_model.add(new_chain)

    new_structure = PDB.Structure.Structure("framework_ref")
    new_structure.add(new_model)
    io = PDB.PDBIO()
    io.set_structure(new_structure)
    io.save(output_path)


def _count_existing_samples(output_dir: str, name: str) -> int:
    """Count PDB files already written to output_dir matching {name}_{int}.pdb."""
    pattern = re.compile(rf"^{re.escape(name)}_(\d+)\.pdb$")
    if not os.path.isdir(output_dir):
        return 0
    return sum(1 for f in os.listdir(output_dir) if pattern.match(f))


def preprocess_csv_and_pkl(args, output_dir: str) -> Optional[str]:
    """Process PDB files and generate pkl and metadata CSV."""
    csv_path = os.path.join(output_dir, f"{args.name}_input.csv")

    start_sample_id = 0
    if getattr(args, 'resume', False):
        already_done = _count_existing_samples(args.output_dir, args.name)
        if already_done >= args.samples_per_target:
            print(f"[resume] All {args.samples_per_target} samples already generated. Nothing to do.")
            return None
        if already_done > 0:
            start_sample_id = already_done
            print(f"[resume] Found {already_done} existing samples; generating {args.samples_per_target - already_done} more (ids {start_sample_id}–{args.samples_per_target - 1}).")
    else:
        # Fresh run: remove any stale CSV from a previous (failed) run to avoid
        # appending to it and producing duplicate rows.
        if os.path.exists(csv_path):
            os.remove(csv_path)

    input_info = {
        "antigen_pdb": args.antigen_pdb,
        "antigen_chain": args.antigen_chain,
        "hotspots": args.specified_hotspots,
        "heavy_chain": args.heavy_chain,
        "pdb_name": args.name,
    }

    scaffold_pdb = getattr(args, "scaffold_pdb", None)
    if scaffold_pdb is not None:
        input_info["scaffold_pdb"] = scaffold_pdb
        input_info["redesign_residues"] = args.scaffold_redesign_residues
        input_info["redesign_lengths"] = args.scaffold_redesign_lengths
        redesign_chains = getattr(args, "scaffold_redesign_chains", None)
        if redesign_chains is not None:
            input_info["redesign_chains"] = redesign_chains
    else:
        input_info["framework_pdb"] = args.framework_pdb
        input_info["cdr_length"] = args.cdr_length

    if hasattr(args, "light_chain") and args.light_chain:
        input_info["light_chain"] = args.light_chain

    for sample_id in range(start_sample_id, args.samples_per_target):
        metadata = process_file(
            input_info, write_dir=output_dir, sample_id=sample_id
        )
        metadata_df = pd.DataFrame([metadata])
        header = sample_id == start_sample_id
        metadata_df.to_csv(csv_path, index=False, mode="a", header=header)

    return csv_path


# =============================================================================
# Main Pipeline
# =============================================================================


def run_pipeline(args):
    """Execute the full sampling pipeline."""
    import json

    # Parse scaffold redesign args from JSON strings (CLI) or leave as-is (pipeline.py Namespace)
    if isinstance(getattr(args, "scaffold_redesign_residues", None), str):
        args.scaffold_redesign_residues = json.loads(args.scaffold_redesign_residues)
    if isinstance(getattr(args, "scaffold_redesign_lengths", None), str):
        args.scaffold_redesign_lengths = json.loads(args.scaffold_redesign_lengths)
    if isinstance(getattr(args, "scaffold_redesign_chains", None), str):
        args.scaffold_redesign_chains = json.loads(args.scaffold_redesign_chains)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Preprocessing stage
    input_data_dir = os.path.join(output_dir, "input")
    os.makedirs(input_data_dir, exist_ok=True)
    processed_csv_path = preprocess_csv_and_pkl(
        args=args, output_dir=input_data_dir
    )
    if processed_csv_path is None:
        return

    # In scaffold mode, generate a framework reference PDB for RMSD validation
    scaffold_pdb = getattr(args, "scaffold_pdb", None)
    if scaffold_pdb is not None:
        framework_ref_path = os.path.join(input_data_dir, f"{args.name}_framework_ref.pdb")
        _generate_framework_reference(
            scaffold_pdb, args.scaffold_redesign_residues,
            args.heavy_chain, framework_ref_path,
            redesign_chains=getattr(args, "scaffold_redesign_chains", None),
            light_chain=getattr(args, "light_chain", None),
        )
        framework_pdb = framework_ref_path
    else:
        framework_pdb = args.framework_pdb

    # Load and update configuration
    conf = ConfigManager(args.config)
    update_configs = {
        "ppi_dataset": {
            "test_csv_path": processed_csv_path,
            "samples_per_target": 1,
            "define_hotspots": True,
        },
        "experiment": {
            "testing_model": {
                "ckpt_path": args.model_weights,
                "save_dir": args.output_dir,
                "antigen_chain": args.antigen_chain,
                "heavy_chain": args.heavy_chain,
                "light_chain": args.light_chain,
            },
            "checkpointer": {"dirpath": args.output_dir},
            "framework_pdb": framework_pdb,
        },
    }
    conf.update_config(update_configs)
    os.makedirs(f"{args.output_dir}/yaml", exist_ok=True)
    config_save_path = os.path.join(
        output_dir, f"yaml/antibody_sample_config_{args.name}.yml"
    )
    conf.save_config(config_save_path)

    # Initialize model
    print("\nInitializing model...")
    t1 = time.time()
    cfg = OmegaConf.create(conf.current_config)
    exp = Experiment(cfg=cfg)

    # Run inference
    print("\nRunning inference...")
    exp.test()

    t2 = time.time()
    print(f"Sample finished in {round(t2 - t1, 2)} seconds")


# =============================================================================
# Command Line Interface
# =============================================================================


def get_parser():
    parser = argparse.ArgumentParser(
        description="Protein Structure Prediction Pipeline"
    )

    # Antigen
    parser.add_argument(
        "--antigen_pdb",
        type=str,
        help="Input antigen protein PDB file path",
    )
    parser.add_argument(
        "--antigen_chain",
        type=str,
        help="Chain id of the antigen chain(s)",
    )
    parser.add_argument(
        "--specified_hotspots",
        type=str,
        help="Specify hotspot residues, e.g., 'C56,C58'",
    )

    # Framework — provide either framework_pdb+cdr_length (old) or
    # scaffold_pdb+scaffold_redesign_residues+scaffold_redesign_lengths (new)
    parser.add_argument(
        "--framework_pdb",
        type=str,
        default=None,
        help="Pre-trimmed framework PDB (CDR loops removed, gaps present).",
    )
    parser.add_argument(
        "--scaffold_pdb",
        type=str,
        default=None,
        help="Complete scaffold PDB (all residues present). "
             "Use with --scaffold_redesign_residues and --scaffold_redesign_lengths.",
    )
    parser.add_argument(
        "--scaffold_redesign_residues",
        type=str,
        default=None,
        help="JSON-style list of inclusive [start,end] residue ranges to redesign, "
             "e.g. '[[26,33],[51,57],[96,110]]'.",
    )
    parser.add_argument(
        "--scaffold_redesign_lengths",
        type=str,
        default=None,
        help="JSON-style list of [min,max] length ranges per redesign region, "
             "e.g. '[[8,8],[8,8],[9,21]]'.",
    )
    parser.add_argument(
        "--scaffold_redesign_chains",
        type=str,
        default=None,
        help="JSON-style list of chain IDs per redesign region, "
             "e.g. '[\"A\",\"A\",\"A\",\"B\",\"B\",\"B\"]'. "
             "If omitted, all regions assigned to heavy_chain.",
    )
    parser.add_argument(
        "--heavy_chain", type=str, help="Chain id of the heavy chain"
    )
    parser.add_argument(
        "--light_chain",
        type=str,
        default=None,
        help="Chain id of the light chain (optional, for antibody)",
    )

    # CDR
    parser.add_argument(
        "--cdr_length",
        type=str,
        default="CDRH1,5-12,CDRH2,4-17,CDRH3,5-26,CDRL1,5-12,CDRL2,3-10,CDRL3,4-13",
        help="Sample CDR length, e.g. 'CDRH1,5-12,CDRH2,4-17,CDRH3,5-26,CDRL1,5-12,CDRL2,3-10,CDRL3,4-13'",
    )

    # Model configs
    parser.add_argument(
        "--config", type=str, default="./configs/test_antibody.yaml"
    )

    # Samples
    parser.add_argument(
        "--samples_per_target",
        type=int,
        default=100,
        help="Number of samples",
    )

    # Checkpoint
    parser.add_argument(
        "--model_weights", type=str, help="Model weights file path"
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--name", type=str, default="test_target", help="Test target name"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume generation, skipping samples already written to output_dir.",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Validate input files
    assert os.path.exists(
        args.antigen_pdb
    ), f"Antigen PDB file not found: {args.antigen_pdb}"
    assert os.path.exists(
        args.config
    ), f"Config file not found: {args.config}"
    assert os.path.exists(
        args.model_weights
    ), f"Model weights file not found: {args.model_weights}"

    antigen_chain = args.antigen_chain
    hotspots = args.specified_hotspots.strip()
    assert (
        hotspots[0] == antigen_chain
    ), "Hotspot chain must be the same as the antigen chain"

    # Mode selection message
    if args.light_chain:
        print("Running Antibody mode (heavy + light chain)")
    else:
        print("Running Nanobody mode (heavy chain only)")

    run_pipeline(args)
