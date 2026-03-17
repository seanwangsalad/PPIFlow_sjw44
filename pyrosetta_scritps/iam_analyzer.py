#!/usr/bin/env python3
"""
Interface Atom/residue energy Analyzer (iam_analyzer.py)

Usage:
    iam_analyzer.py <file.pdb|file.cif> <chain_binder> [options]
    iam_analyzer.py structure.pdb A
    iam_analyzer.py structure.pdb A,B --cutoff 5.0 -o my_plot.png
"""
import sys
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from Bio.PDB import PDBParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pr_utils.biopython_utils import hotspot_residues
from pr_utils.pyrosetta_utils import convert_cif_to_pdb, energy_interacting_residues, fastrelax


def get_all_chains(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)
    return [chain.id for chain in structure[0]]


def get_interface_energies_for_chain(pdb_path, binder_chain, target_chains, cutoff):
    """
    For one binder chain, find all interface residues and their summed
    Rosetta energies against all target chains.

    Returns:
        energies: dict {resnum -> total_energy}
        labels:   dict {resnum -> single_letter_code}
    """
    # Per-binder-residue data, aggregated across all target chains
    combined_contacts = {}   # resnum -> {"res": aa, "contacts": set()}
    energies = {}            # resnum -> summed energy

    for target_chain in target_chains:
        contacts = hotspot_residues(pdb_path, target_chain, binder_chain, cutoff)
        if not contacts:
            continue

        # Merge residue labels
        for resnum, data in contacts.items():
            if resnum not in combined_contacts:
                combined_contacts[resnum] = {"res": data["res"], "contacts": set()}
            combined_contacts[resnum]["contacts"].update(data["contacts"])

        # Compute per-residue energies against this target chain
        chain_energies = energy_interacting_residues(
            pdb_path, target_chain, binder_chain, contacts
        )
        for resnum, e in chain_energies.items():
            energies[resnum] = energies.get(resnum, 0.0) + e

    labels = {r: d["res"] for r, d in combined_contacts.items()}
    return energies, labels


def plot_interface_energies(energy_data, label_data, binder_chains, input_file, output_path):
    n = len(binder_chains)
    fig, axes = plt.subplots(n, 1, figsize=(14, 5 * n), squeeze=False)
    fig.suptitle(
        f"Interface Residue Energies\n{os.path.basename(input_file)}",
        fontsize=13, fontweight='bold', y=1.01 if n > 1 else 1.02
    )

    pos_color = '#d73027'   # red  – unfavorable
    neg_color = '#4575b4'   # blue – favorable

    for i, chain in enumerate(binder_chains):
        ax = axes[i][0]
        energies = energy_data[chain]
        labels = label_data[chain]

        if not energies:
            ax.text(0.5, 0.5,
                    f'No interface residues found for binder chain {chain}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.set_title(f'Binder Chain {chain}', fontsize=12, fontweight='bold')
            continue

        sorted_resnums = sorted(energies.keys())
        x_labels = [f"{labels.get(r, '?')}{r}" for r in sorted_resnums]
        y_values  = [energies[r] for r in sorted_resnums]
        x_pos     = np.arange(len(sorted_resnums)) * 0.55

        colors = [pos_color if v >= 0 else neg_color for v in y_values]
        bars = ax.bar(x_pos, y_values, color=colors, edgecolor='black',
                      linewidth=0.4, width=0.4)

        # Expand y limits to make room for annotations
        y_range = max(y_values) - min(y_values) if max(y_values) != min(y_values) else 1.0
        ax.set_ylim(min(y_values) - 0.18 * y_range, max(y_values) + 0.18 * y_range)

        # Annotate each bar with its energy value
        for bar, val in zip(bars, y_values):
            offset = 0.02 * y_range
            va = 'bottom' if val >= 0 else 'top'
            y_text = val + offset if val >= 0 else val - offset
            ax.text(bar.get_x() + bar.get_width() / 2, y_text,
                    f'{val:.2f}', ha='center', va=va, fontsize=6.5, rotation=0,
                    fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=60, ha='right', fontsize=8)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_ylabel('Rosetta Energy (REU)', fontsize=10)
        ax.set_xlabel('Binder Interface Residues', fontsize=10)
        ax.set_title(f'Binder Chain {chain}  –  {len(sorted_resnums)} interface residues',
                     fontsize=12, fontweight='bold')
        ax.yaxis.grid(True, alpha=0.3, linestyle=':')
        ax.set_axisbelow(True)

        # Legend
        legend_handles = [
            mpatches.Patch(color=neg_color, label='Favorable (< 0 REU)'),
            mpatches.Patch(color=pos_color, label='Unfavorable (≥ 0 REU)'),
        ]
        ax.legend(handles=legend_handles, fontsize=8, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Interface Atom/residue energy Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('file',
                        help='Input PDB or CIF structure file')
    parser.add_argument('chain_binder',
                        help='Binder chain(s), comma-separated (e.g. "A" or "A,B")')
    parser.add_argument('--cutoff', type=float, default=4.0,
                        help='Heavy-atom distance cutoff in Å for interface detection (default: 4.0)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output plot filename (default: <input_base>_interface_energies.png)')
    parser.add_argument('--relax', action='store_true',
                        help='Run FastRelax on the structure before analysis')
    args = parser.parse_args()

    input_file    = args.file
    binder_chains = [c.strip() for c in args.chain_binder.split(',') if c.strip()]

    if not os.path.isfile(input_file):
        print(f"Error: file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    # Convert CIF → PDB if needed
    if input_file.lower().endswith('.cif'):
        print(f"Converting CIF → PDB: {input_file}")
        pdb_path = convert_cif_to_pdb(input_file)
        print(f"  Written: {pdb_path}")
    else:
        pdb_path = input_file

    # FastRelax before analysis (optional)
    if args.relax:
        print("Running FastRelax…")
        relaxed_pose = fastrelax(pdb_path)
        relaxed_pdb  = os.path.splitext(pdb_path)[0] + "_relaxed.pdb"
        relaxed_pose.dump_pdb(relaxed_pdb)
        print(f"  Relaxed structure written: {relaxed_pdb}")
        pdb_path = relaxed_pdb

    # Determine target chains
    all_chains    = get_all_chains(pdb_path)
    target_chains = [c for c in all_chains if c not in binder_chains]

    print(f"All chains detected : {all_chains}")
    print(f"Binder chain(s)     : {binder_chains}")
    print(f"Target chain(s)     : {target_chains}")

    if not target_chains:
        print("Error: no target chains remain after selecting binder chain(s). "
              "Check your chain_binder argument.", file=sys.stderr)
        sys.exit(1)

    # Compute per-residue interface energies for each binder chain
    energy_data = {}
    label_data  = {}

    for chain in binder_chains:
        print(f"\n[Chain {chain}] Computing interface energies (cutoff={args.cutoff} Å)…")
        energies, labels = get_interface_energies_for_chain(
            pdb_path, chain, target_chains, args.cutoff
        )
        energy_data[chain] = energies
        label_data[chain]  = labels

        if energies:
            total = sum(energies.values())
            print(f"  Interface residues : {len(energies)}")
            print(f"  Total energy       : {total:.3f} REU")
            favorable   = sum(v for v in energies.values() if v < 0)
            unfavorable = sum(v for v in energies.values() if v >= 0)
            print(f"  Favorable sum      : {favorable:.3f} REU")
            print(f"  Unfavorable sum    : {unfavorable:.3f} REU")
        else:
            print("  No interface residues detected.")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base        = os.path.splitext(input_file)[0]
        output_path = f"{base}_interface_energies.png"

    plot_interface_energies(energy_data, label_data, binder_chains, input_file, output_path)


if __name__ == '__main__':
    main()
