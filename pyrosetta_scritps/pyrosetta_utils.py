import os
import math
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
from Bio import BiopythonWarning
from Bio.PDB import PDBParser, DSSP, Selection, Polypeptide, PDBIO, Select, Chain, Superimposer
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB.Selection import unfold_entities
from Bio.PDB.Polypeptide import is_aa
import pyrosetta as pr
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector, OrResidueSelector
from pyrosetta.rosetta.core.kinematics import MoveMap


options = """
-ignore_unrecognized_res
-include_sugars
-auto_detect_glycan_connections
-maintain_links 
-alternate_3_letter_codes pdb_sugar
-write_glycan_pdb_codes
-ignore_zero_occupancy false 
-load_PDB_components false
-no_fconfig
-mute all
"""
pr.init(" ".join(options.split('\n')), silent=True)

def convert_cif_to_pdb(cif_path):
    pdb_path = cif_path.replace(".cif", ".pdb")
    pose = pr.pose_from_file(cif_path)
    pose.dump_pdb(pdb_path)
    return pdb_path

def fastrelax(traj, s_r=3, iter=150):
    if isinstance(traj, pr.Pose):
        pose = traj
    else:
        pose = pr.pose_from_pdb(traj)
    
    movemap = MoveMap()
    movemap.set_chi(True)
    movemap.set_bb(True)
    movemap.set_jump(False)

    fast_relax = FastRelax(standard_repeats=s_r)
    scorefxn = pr.get_fa_scorefxn()
    fast_relax.set_scorefxn(scorefxn)
    fast_relax.set_movemap(movemap)
    fast_relax.max_iter(iter)
    fast_relax.min_type("lbfgs_armijo_nonmonotone")
    fast_relax.constrain_relax_to_start_coords(True)

    fast_relax.apply(pose)
    return pose


def energy_interacting_residues(traj, target_chain, binder_chain, interacting_residues):
    if isinstance(traj, pr.Pose):
        pose = traj
    else:
        pose = pr.pose_from_pdb(traj)

    scorefxn = pr.get_fa_scorefxn()
    scorefxn(pose)
    pdb_info = pose.pdb_info()
    energy_graph = pose.energies().energy_graph()

    results_dg = {}
    for binder_res, data in interacting_residues.items():
        b_idx = pdb_info.pdb2pose(binder_chain, binder_res)
        t_indices = [pdb_info.pdb2pose(target_chain, r)
                     for r in data['contacts']
                     if pdb_info.pdb2pose(target_chain, r) != 0]

        if b_idx == 0 or not t_indices:
            continue

        dg = 0.0
        for t_idx in t_indices:
            edge = energy_graph.find_edge(b_idx, t_idx)
            if edge is not None:
                dg += edge.dot(scorefxn.weights())

        results_dg[binder_res] = dg

    return results_dg


