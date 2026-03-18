####################################
################ BioPython functions
####################################
### Import dependencies
import os
import math
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser, DSSP, Selection



three_to_one_map = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# identify interacting residues at the binder interface
def hotspot_residues(trajectory_pdb, target_chain, binder_chain, atom_distance_cutoff=4.0):
    # Parse the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", trajectory_pdb)

    # Get the specified chain
    binder_atoms = Selection.unfold_entities(structure[0][binder_chain], 'A')
    binder_coords = np.array([atom.coord for atom in binder_atoms])

    # Get atoms and coords for the target chain
    target_atoms = Selection.unfold_entities(structure[0][target_chain], 'A')
    target_coords = np.array([atom.coord for atom in target_atoms])

    # Build KD trees for both chains
    binder_tree = cKDTree(binder_coords)
    target_tree = cKDTree(target_coords)

    # Prepare to collect interacting residues
    interacting_residues = {}

    # Query the tree for pairs of atoms within the distance cutoff
    pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)

    # Process each binder atom's interactions
    for binder_idx, close_indices in enumerate(pairs):
        if not close_indices:
            continue
        
        binder_residue = binder_atoms[binder_idx].get_parent()
        binder_resname = binder_residue.get_resname()

        # Convert three-letter code to single-letter code using the manual dictionary
        if binder_resname in three_to_one_map:
            aa_single_letter = three_to_one_map[binder_resname]
            binder_resnum = binder_residue.id[1]
            if binder_resnum not in interacting_residues:
                interacting_residues[binder_resnum] = {"res": aa_single_letter, "contacts": set()}
            for close_idx in close_indices:
                target_residue = target_atoms[close_idx].get_parent()
                interacting_residues[binder_resnum]["contacts"].add(target_residue.id[1])

    return interacting_residues

def secondary_structure(trajectory_pdb, chain):
    # Parse the structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', trajectory_pdb)
    model = structure[0]

    # Initialize DSSP - ensure 'mkdssp' is in your path
    try:
        dssp = DSSP(model, trajectory_pdb, dssp="mkdssp")
    except Exception as e:
        print(f"DSSP Error: {e}")
        return [], [], []

    helices, sheets, loops = [], [], []
    
    # Access keys directly to avoid the 'items' attribute error
    dssp_keys = list(dssp.keys())
    
    for key in dssp_keys:
        chain_id = key[0]

        if chain_id != chain:
            continue

        # Extract data using the key
        res_info = dssp[key]
        res_num  = key[1][1] # PDB residue number
        ss_code  = res_info[2] # Secondary structure code

        if ss_code in ('H', 'G', 'I'): # Helices
            helices.append(res_num)
        elif ss_code in ('E', 'B'):    # Sheets
            sheets.append(res_num)
        else:                          # Loops (T, S, -, etc.)
            loops.append(res_num)

    return helices, sheets, loops
