import os
import json
import shutil
import subprocess
import tempfile
import zipfile
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import requests
import re

from ase.io import read, write
from jarvis.core.atoms import Atoms, ase_to_atoms
from jarvis.db.figshare import get_jid_data
from jarvis.io.vasp.outputs import Vasprun


def extract_kpoints_from_vasprun(vasprun_file):
    """Extract k-point list from vasprun.xml."""
    with open(vasprun_file, "r") as file:
        lines = file.readlines()

    start = next(
        (i for i, line in enumerate(lines) if "kpointlist" in line.lower()),
        None,
    )
    end = next(
        (
            i
            for i, line in enumerate(lines[start:], start)
            if "</varray>" in line.lower()
        ),
        None,
    )

    if start is None or end is None:
        raise ValueError("Could not find k-point list in vasprun.xml.")

    kpoints = []
    for line in lines[start + 1 : end]:
        coords = [
            float(x) for x in line.strip().strip("<v>").strip("</v>").split()
        ]
        kpoints.append(coords)

    return kpoints


class DFTBCalculator:
    """Simplified DFTB+ calculator for geometry optimization and band structure calculations."""

    def __init__(self, dftb_executable, sk_dir, k_mesh=[8, 8, 8]):
        self.dftb_executable = dftb_executable
        self.sk_dir = sk_dir
        self.k_mesh = k_mesh

        # Check if executable exists
        if not os.path.exists(dftb_executable):
            raise FileNotFoundError(
                f"DFTB+ executable not found: {dftb_executable}"
            )

        # Check if SK files directory exists
        if not os.path.exists(sk_dir):
            raise FileNotFoundError(
                f"Slater-Koster files directory not found: {sk_dir}"
            )

    def run_command(self, cmd, cwd=None, log="run.log", text=True):
        """Run command and stream output."""
        print(f"Running: {' '.join(cmd)}")
        log_path = Path(cwd) / log if cwd else log

        with open(log_path, "a") as fh:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=text,
            )
            for line in proc.stdout:
                print(line.rstrip())
                fh.write(line)

        if proc.wait() != 0:
            raise RuntimeError(f"Command {' '.join(cmd)} failed.")

    def get_max_angular_momentum(self, atoms):
        """Get MaxAngularMomentum block for elements."""
        lmap = {}
        for sym in sorted(set(atoms.get_chemical_symbols())):
            Z = atoms[atoms.get_chemical_symbols().index(sym)].number
            if Z == 1:
                lmap[sym] = "s"  # H
            elif Z <= 10:
                lmap[sym] = "p"  # He–Ne
            elif Z <= 56:
                lmap[sym] = "d"  # transition metals
            else:
                lmap[sym] = "f"  # lanthanides/actinides

        return "\n".join(f'    {el} = "{l}"' for el, l in lmap.items())

    def create_scf_input(self, atoms, output_file, optimize=True):
        """Create SCF calculation input file."""
        maxl_block = self.get_max_angular_momentum(atoms)

        hsd_content = f"""Geometry = GenFormat {{
    <<< "geo.gen"
}}

Hamiltonian = DFTB {{
    SCC = Yes
    MaxAngularMomentum = {{
{maxl_block}
    }}
    SlaterKosterFiles = Type2FileNames {{
        Prefix = "{self.sk_dir}/"
        Separator = "-"
        Suffix = ".skf"
    }}
    KPointsAndWeights = SuperCellFolding {{
        {self.k_mesh[0]} 0 0
        0 {self.k_mesh[1]} 0
        0 0 {self.k_mesh[2]}
        0.0 0.0 0.0
    }}
    Filling = Fermi {{
        Temperature [Kelvin] = 0
    }}
}}

Analysis = {{
    CalculateForces = Yes
    ProjectStates = {{
        Region = {{
            Atoms = 1:-1
            OrbitalResolved = No
        }}
    }}
}}

Options = {{
    WriteResultsTag = Yes
}}

ParserOptions = {{
    ParserVersion = 12
}}"""

        if optimize:
            driver_block = """
Driver = GeometryOptimization {
    Optimizer = Rational {}
    MovedAtoms = 1:-1
    MaxSteps = 100
    OutputPrefix = "geom.out"
    Convergence = {
        GradElem = 1E-4
    }
    LatticeOpt = Yes
}"""
            hsd_content = hsd_content.replace(
                "Geometry =", driver_block + "\n\nGeometry ="
            )

        with open(output_file, "w") as f:
            f.write(hsd_content)

    def create_band_input(self, atoms, kpoints, output_file):
        """Create band structure calculation input file."""
        maxl_block = self.get_max_angular_momentum(atoms)

        # Format k-points for band structure
        kpoint_lines = []
        for kp in kpoints:
            kpoint_lines.append(
                f"    1  {kp[0]:.8f}  {kp[1]:.8f}  {kp[2]:.8f}"
            )
        kpoint_block = "\n".join(kpoint_lines)

        hsd_content = f"""Geometry = GenFormat {{
    <<< "geom.out.gen"
}}

Hamiltonian = DFTB {{
    SCC = Yes
    ReadInitialCharges = Yes
    MaxAngularMomentum = {{
{maxl_block}
    }}
    SlaterKosterFiles = Type2FileNames {{
        Prefix = "{self.sk_dir}/"
        Separator = "-"
        Suffix = ".skf"
    }}
    KPointsAndWeights = Klines {{
{kpoint_block}
    }}
}}

Options = {{
    WriteResultsTag = Yes
}}

ParserOptions = {{
    ParserVersion = 12
}}"""

        with open(output_file, "w") as f:
            f.write(hsd_content)

    def extract_results(self, detailed_out_path):
        """Extract energy, forces, and Fermi level from detailed.out."""
        energy = None
        forces = []
        fermi_ev = None

        with open(detailed_out_path, "r") as f:
            lines = f.readlines()

        # Extract energy
        for line in lines:
            if "Total energy:" in line:
                energy = float(line.split()[-2]) * 27.2114  # Ha to eV
                break

        # Extract forces
        for i, line in enumerate(lines):
            if "Total Forces" in line:
                for l in lines[i + 2 :]:
                    if not l.strip():
                        break
                    parts = l.split()
                    forces.append(
                        [float(x) * 51.422086 for x in parts[-3:]]
                    )  # Ha/Bohr to eV/Å
                break

        # Extract Fermi level
        for line in lines:
            if "Fermi" in line and "eV" in line:
                match = re.search(r"([-+]?\d*\.\d+|\d+)\s*eV", line)
                if match:
                    fermi_ev = float(match.group(1))
                break

        return energy, np.array(forces), fermi_ev

    def calculate_dos(self, work_dir, fermi_ev):
        """Calculate DOS data."""
        band_out = work_dir / "band.out"
        dos_file = work_dir / "dos_total.dat"

        # Generate DOS using dp_dos
        os.system(f"dp_dos {band_out} {dos_file}")

        # Read and process DOS data
        x_values, y_values = [], []
        with open(dos_file, "r") as f:
            for line in f:
                tokens = line.split()
                if tokens and self._is_number(tokens[0]):
                    x_values.append(float(tokens[0]))
                    y_values.append(float(tokens[1]))

        # Convert to numpy arrays and adjust for Fermi level
        energy = np.array(x_values) - fermi_ev
        dos = np.array(y_values)

        # Reshape for multiple bands if necessary
        num_bands = len(y_values) // len(x_values)
        if len(y_values) % len(x_values) == 0 and num_bands > 1:
            dos = dos.reshape(num_bands, len(x_values))

        return energy, dos

    def calculate_band_properties(self, band_energies):
        """Calculate band properties: bandgap, VBM, CBM."""
        # Find the highest occupied band (VBM) and lowest unoccupied band (CBM)
        vbm = -float("inf")
        cbm = float("inf")

        for kpt_bands in band_energies:
            for energy in kpt_bands:
                if energy <= 0:  # Below Fermi level (valence bands)
                    vbm = max(vbm, energy)
                else:  # Above Fermi level (conduction bands)
                    cbm = min(cbm, energy)

        # Handle metallic systems
        if vbm == -float("inf") or cbm == float("inf"):
            bandgap = 0.0
            if vbm == -float("inf"):
                vbm = None
            if cbm == float("inf"):
                cbm = None
        else:
            bandgap = max(0.0, cbm - vbm)

        return bandgap, vbm, cbm

    def create_band_structure_plot(self, kpoints, energies, output_file):
        """Create band structure plot and return plot data."""
        plt.figure(figsize=(10, 6))

        # Plot each band
        for i in range(energies.shape[1]):
            plt.plot(kpoints, energies[:, i], "b-", linewidth=1.5, alpha=0.8)

        # Add Fermi level line
        plt.axhline(
            y=0, color="r", linestyle="--", linewidth=1, label="Fermi Level"
        )

        # Formatting
        plt.xlabel("K-point Path")
        plt.ylabel("Energy (E - E_F) [eV]")
        plt.title("Band Structure")
        plt.ylim(-6, 6)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # Save plot
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        # Return plot data for potential reuse
        plot_data = {
            "kpoints": kpoints.tolist(),
            "energies": energies.tolist(),
            "fermi_level": 0.0,
            "energy_range": [-6, 6],
            "plot_file": str(output_file),
        }

        return plot_data

    def create_dos_plot(self, energy, dos, output_file):
        """Create DOS plot and return plot data."""
        plt.figure(figsize=(8, 6))

        # Plot DOS
        if dos.ndim == 2:
            for i in range(dos.shape[0]):
                plt.plot(energy, dos[i, :], "k-", linewidth=1.5, alpha=0.8)
        else:
            plt.plot(energy, dos, "k-", linewidth=1.5)

        # Add Fermi level line
        plt.axvline(
            x=0, color="r", linestyle="--", linewidth=1, label="Fermi Level"
        )

        # Formatting
        plt.xlabel("Energy (E - E_F) [eV]")
        plt.ylabel("DOS [states/eV]")
        plt.title("Density of States")
        plt.xlim(-6, 6)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # Save plot
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        # Return plot data
        plot_data = {
            "energy": energy.tolist(),
            "dos": dos.tolist() if dos.ndim == 1 else dos.tolist(),
            "fermi_level": 0.0,
            "energy_range": [-6, 6],
            "plot_file": str(output_file),
        }

        return plot_data

    def save_electronic_properties(
        self, work_dir, energy, fermi_ev, bandgap=None, vbm=None, cbm=None
    ):
        """Save electronic properties to JSON file."""
        properties = {
            "total_energy_eV": energy,
            "fermi_level_eV": fermi_ev,
            "bandgap_eV": bandgap,
            "vbm_eV": vbm,
            "cbm_eV": cbm,
        }

        # Remove None values
        properties = {k: v for k, v in properties.items() if v is not None}

        with open(Path(work_dir) / "electronic_properties.json", "w") as f:
            json.dump(properties, f, indent=2)

        return properties

    def compare_with_vasp(
        self, work_dir, vasprun, output_file="comparison.png"
    ):
        """Compare DFTB+ band structure with VASP."""
        # Read DFTB+ band data
        band_file = work_dir / "band_tot.dat"
        os.system(f"dp_bands {work_dir}/band.out {work_dir}/band")

        dftb_data = pd.read_csv(band_file, delim_whitespace=True, header=None)
        dftb_kpts = dftb_data.iloc[:, 0].values
        dftb_bands = dftb_data.iloc[:, 1:].values

        # Get Fermi level and adjust
        _, _, fermi_ev = self.extract_results(work_dir / "detailed.out")
        dftb_bands -= fermi_ev

        # Process VASP data
        vasp_fermi = vasprun.efermi
        vasp_eigs = (
            np.array([eig[:, 0] for eig in vasprun.eigenvalues[0]]).T
            - vasp_fermi
        )

        # Compare bands
        energy_tol = 4
        differences = []

        for k in range(min(dftb_bands.shape[0], vasp_eigs.shape[1])):
            for dftb_e in dftb_bands[k]:
                if -energy_tol < dftb_e < energy_tol:
                    vasp_band = vasp_eigs[:, k]
                    valid_vasp = vasp_band[
                        (vasp_band > -energy_tol) & (vasp_band < energy_tol)
                    ]
                    if len(valid_vasp) > 0:
                        min_diff = np.min(np.abs(dftb_e - valid_vasp))
                        differences.append(min_diff)

        max_diff = max(differences) if differences else 0

        # Plot comparison
        fig = plt.figure(figsize=(12, 5))
        gs = GridSpec(1, 2)

        plt.subplot(gs[0, 0])
        plt.title("(a) Band Structure Comparison")
        for band in dftb_bands.T:
            plt.plot(
                band,
                "b-",
                alpha=0.7,
                label=(
                    "DFTB+"
                    if "DFTB+" not in plt.gca().get_legend_handles_labels()[1]
                    else ""
                ),
            )
        for band in vasp_eigs:
            plt.plot(
                band,
                "r-",
                alpha=0.7,
                label=(
                    "VASP"
                    if "VASP" not in plt.gca().get_legend_handles_labels()[1]
                    else ""
                ),
            )
        plt.ylim(-energy_tol, energy_tol)
        plt.ylabel("Energy [eV]")
        plt.legend()

        plt.subplot(gs[0, 1])
        plt.title("(b) Error Distribution")
        plt.scatter(
            differences,
            [dftb_bands.flatten()[i] for i in range(len(differences))],
            alpha=0.6,
        )
        plt.ylim(-energy_tol, energy_tol)
        plt.xlabel("DFTB - VASP (δ) [eV]")
        plt.ylabel("Energy [eV]")

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return max_diff

    def run_optimization(self, atoms, work_dir="opt"):
        """Run geometry optimization calculation."""
        work_path = Path(work_dir)
        work_path.mkdir(exist_ok=True)

        # Write geometry and create input
        geo_file = work_path / "geo.gen"
        write(geo_file, atoms, format="gen")
        self.create_scf_input(atoms, work_path / "dftb_in.hsd", optimize=True)
        name = work_path / "geom.out.gen"
        if not os.path.exists(name):
            # Run calculation
            self.run_command([self.dftb_executable], cwd=work_path)
        else:
            print(name, "exists already")
        # Extract results
        energy, forces, fermi_ev = self.extract_results(
            work_path / "detailed.out"
        )
        final_atoms = read(work_path / "geom.out.gen")

        # Calculate DOS for optimization
        energy_grid, dos = self.calculate_dos(work_path, fermi_ev)
        dos_plot_data = self.create_dos_plot(
            energy_grid, dos, work_path / "dos.png"
        )

        band_file = work_path / "band_tot.dat"
        os.system(f"dp_bands {work_path}/band.out {work_path}/band")

        # Read band structure data
        band_data = pd.read_csv(band_file, delim_whitespace=True, header=None)
        band_kpts = band_data.iloc[:, 0].values
        band_energies = band_data.iloc[:, 1:].values

        # Adjust energies relative to Fermi level
        band_energies_adjusted = band_energies - fermi_ev

        # Calculate band properties
        # bandgap, vbm, cbm = self.calculate_band_properties(band_energies_adjusted)

        # Read the band.out file
        band_data = pd.read_csv(
            f"{work_path}/band.out", sep="\s+", header=None, comment="#"
        )

        # Remove the header line (e.g., rows starting with 'KPT')
        band_data = band_data[
            pd.to_numeric(band_data[0], errors="coerce").notna()
        ]

        # Convert energy and occupation columns
        band_data[1] = band_data[1].astype(float)  # Energy
        band_data[2] = band_data[2].astype(float)  # Occupation

        # VBM: max energy where occupation > 0
        vbm = band_data[1][band_data[2] > 0].max()

        # CBM: min energy where occupation == 0
        cbm = band_data[1][band_data[2] == 0].min()

        # Band gap
        bandgap = cbm - vbm if cbm > vbm else 0.0

        print(f"New VBM: {vbm:.4f} eV")
        print(f"New CBM: {cbm:.4f} eV")
        print(f"New Band Gap: {bandgap:.4f} eV")

        # Save electronic properties JSON
        electronic_props = self.save_electronic_properties(
            work_path, energy, fermi_ev
        )

        results = {
            "energy": energy,
            "forces": forces.tolist(),
            "fermi_level": fermi_ev,
            "initial_atoms": ase_to_atoms(atoms).to_dict(),
            "final_atoms": ase_to_atoms(final_atoms).to_dict(),
            "electronic_properties": electronic_props,
            "bandgap": bandgap,
            "vbm": vbm,
            "cbm": cbm,
            "dos": {
                "energy_grid": energy_grid.tolist(),
                "dos_values": dos.tolist() if dos.ndim == 1 else dos.tolist(),
                "plot_data": dos_plot_data,
            },
        }
        print("Initial structure", atoms)
        print("Final structure", final_atoms)
        with open(work_path / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        return results, final_atoms

    def run_band_structure(
        self, atoms, kpoints, vasprun=None, work_dir="band"
    ):
        """Run band structure calculation with comprehensive analysis."""
        work_path = Path(work_dir)
        work_path.mkdir(exist_ok=True)

        # Copy required files from optimization
        shutil.copy("opt/charges.bin", work_path / "charges.bin")
        shutil.copy("opt/geom.out.gen", work_path / "geom.out.gen")

        # Create input and run
        self.create_band_input(atoms, kpoints, work_path / "dftb_in.hsd")
        self.run_command([self.dftb_executable], cwd=work_path)

        # Extract basic results
        energy, forces, fermi_ev = self.extract_results(
            work_path / "detailed.out"
        )

        # Generate band structure data
        band_file = work_path / "band_tot.dat"
        os.system(f"dp_bands {work_path}/band.out {work_path}/band")

        # Read band structure data
        band_data = pd.read_csv(band_file, delim_whitespace=True, header=None)
        band_kpts = band_data.iloc[:, 0].values
        band_energies = band_data.iloc[:, 1:].values

        # Adjust energies relative to Fermi level
        band_energies_adjusted = band_energies - fermi_ev

        # Calculate band properties
        # bandgap, vbm, cbm = self.calculate_band_properties(band_energies_adjusted)

        # Read the band.out file
        band_data = pd.read_csv(
            f"{work_path}/band.out", sep="\s+", header=None, comment="#"
        )

        # Remove the header line (e.g., rows starting with 'KPT')
        band_data = band_data[
            pd.to_numeric(band_data[0], errors="coerce").notna()
        ]

        # Convert energy and occupation columns
        band_data[1] = band_data[1].astype(float)  # Energy
        band_data[2] = band_data[2].astype(float)  # Occupation

        # VBM: max energy where occupation > 0
        vbm = band_data[1][band_data[2] > 0].max()

        # CBM: min energy where occupation == 0
        cbm = band_data[1][band_data[2] == 0].min()

        # Band gap
        bandgap = cbm - vbm if cbm > vbm else 0.0

        print(f"New VBM: {vbm:.4f} eV")
        print(f"New CBM: {cbm:.4f} eV")
        print(f"New Band Gap: {bandgap:.4f} eV")

        # Generate DOS data for band directory
        energy_grid, dos_data = self.calculate_dos(work_path, fermi_ev)

        # Create band structure plot (only in band directory)
        band_plot_data = self.create_band_structure_plot(
            band_kpts, band_energies_adjusted, work_path / "band_structure.png"
        )

        # Create DOS plot (in band directory)
        dos_plot_data = self.create_dos_plot(
            energy_grid, dos_data, work_path / "dos.png"
        )

        # Save electronic properties JSON
        electronic_props = self.save_electronic_properties(
            work_path, energy, fermi_ev, bandgap, vbm, cbm
        )

        # Compare with VASP if provided
        max_diff = None
        vasp_comparison = None
        if vasprun:
            max_diff = self.compare_with_vasp(
                work_path, vasprun, work_path / "comparison.png"
            )
            vasp_comparison = {
                "max_difference": max_diff,
                "comparison_plot": str(work_path / "comparison.png"),
            }

        # Compile comprehensive results
        results = {
            "energy": energy,
            "fermi_level": fermi_ev,
            "bandgap": bandgap,
            "vbm": vbm,
            "cbm": cbm,
            "electronic_properties": electronic_props,
            "band_structure": {
                "kpoints": band_kpts.tolist(),
                "energies": band_energies_adjusted.tolist(),
                "plot_data": band_plot_data,
            },
            "dos": {
                "energy_grid": energy_grid.tolist(),
                "dos_values": (
                    dos_data.tolist()
                    if dos_data.ndim == 1
                    else dos_data.tolist()
                ),
                "plot_data": dos_plot_data,
            },
            "vasp_comparison": vasp_comparison,
            "max_difference": max_diff,
        }

        # Save results
        with open(work_path / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        return results

    @staticmethod
    def _is_number(s):
        """Check if string represents a number."""
        try:
            float(s)
            return True
        except ValueError:
            return False

    def run_phonon(
        self,
        atoms,
        work_dir="phonon",
        supercell=[2, 2, 2],
        amplitude=5e-4,
        tolerance=1e-4,
        gamma_centered=False,
    ):
        """
        Run phonon calculation using Phonopy with DFTB+.

        Parameters:
        -----------
        atoms : ASE Atoms object
            Optimized atomic structure
        work_dir : str
            Working directory for phonon calculations
        supercell : list
            Supercell dimensions [nx, ny, nz]
        amplitude : float
            Displacement amplitude for finite differences
        tolerance : float
            Tolerance for symmetry detection
        gamma_centered : bool
            Whether to use gamma-centered k-mesh

        Returns:
        --------
        dict : Phonon calculation results including band structure and DOS
        """
        work_path = Path(work_dir)
        work_path.mkdir(exist_ok=True)

        # Write initial geometry
        geo_file = work_path / "geo.gen"
        write(geo_file, atoms, format="gen")

        print("Setting up phonon calculation with Phonopy...")

        # Generate displaced structures
        dim_str = " ".join(map(str, supercell))
        phonopy_cmd = [
            "phonopy",
            "-d",
            f"--dim={dim_str}",
            f"--amplitude={amplitude}",
            f"--tolerance={tolerance}",
            "--dftb+",
        ]
        print("phonopy_cmd", phonopy_cmd)
        self.run_command(phonopy_cmd, cwd=work_path)

        # Find all displacement files
        disp_files = list(work_path.glob("geo.genS-*"))
        print(f"Found {len(disp_files)} displacement structures")
        if not disp_files:
            raise RuntimeError("No displacement files generated by Phonopy")

        # Calculate forces for each displacement
        force_files = []
        for disp_file in disp_files:
            disp_num = disp_file.name[
                -3:
            ]  # Get last 3 characters (displacement number)
            disp_dir = work_path / f"disp-{disp_num}"
            disp_dir.mkdir(exist_ok=True)

            print(f"Processing displacement {disp_num}...")

            # Copy displacement geometry
            shutil.copy(disp_file, disp_dir / "geo.gen")

            print("HERE 1")
            # Create DFTB+ input for force calculation
            shift = (0.5, 0.5, 0.0) if gamma_centered else (0.0, 0.0, 0.0)
            self.create_force_input(atoms, disp_dir / "dftb_in.hsd", shift)
            print("HERE 2")

            # Run DFTB+ calculation
            self.run_command([self.dftb_executable], cwd=disp_dir)

            # Check if results.tag exists (Phonopy needs this format)
            results_tag = disp_dir / "results.tag"
            print("cwd", os.getcwd())
            print("results_tag", results_tag)
            if not results_tag.exists():
                # Convert detailed.out to results.tag format if needed
                self._convert_to_results_tag(disp_dir)
            pth = Path(str(results_tag))
            force_files.append(str(Path(*pth.parts[-2:])))
            # force_files.append(str(results_tag))

        # Process forces with Phonopy
        print("HERE 3")
        print("Processing forces with Phonopy...", os.getcwd())
        res_tag = " ".join(list(disp_dir.glob("disp-*/results.tag")))
        print("res_tag", res_tag, force_files)
        # phonopy_force_cmd = ["phonopy", "-f"] + rest_tag + ["--dftb+"]

        phonopy_force_cmd = ["phonopy", "-f"] + force_files + ["--dftb+"]
        # phonopy_force_cmd = ["phonopy", "-f"] + map(str,disp_dir.glob("disp-*/results.tag")) + ["--dftb+"]
        print("HERE 4", phonopy_force_cmd)
        self.run_command(phonopy_force_cmd, cwd=work_path)

        # run(["phonopy","-f",*map(str,phdir.glob("disp-*/results.tag")),"--dftb+"],cwd=phdir)

        # Calculate phonon band structure and DOS
        print("Calculating phonon properties...")
        results = self._calculate_phonon_properties(work_path, atoms)

        # Save results
        with open(work_path / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=self._json_serializable)

        print(f"Phonon calculation completed. Results saved in {work_path}/")
        return results

    def create_force_input(self, atoms, output_file, shift=(0.0, 0.0, 0.0)):
        """Create DFTB+ input file for force calculations (no optimization)."""
        maxl_block = self.get_max_angular_momentum(atoms)

        hsd_content = f"""Geometry = GenFormat {{
        <<< "geo.gen"
    }}

    Hamiltonian = DFTB {{
        SCC = Yes
        MaxAngularMomentum = {{
    {maxl_block}
        }}
        SlaterKosterFiles = Type2FileNames {{
            Prefix = "{self.sk_dir}/"
            Separator = "-"
            Suffix = ".skf"
        }}
        KPointsAndWeights = SuperCellFolding {{
            {self.k_mesh[0]} 0 0
            0 {self.k_mesh[1]} 0
            0 0 {self.k_mesh[2]}
            {shift[0]} {shift[1]} {shift[2]}
        }}
        Filling = Fermi {{
            Temperature [Kelvin] = 100
        }}
    }}

    Analysis = {{
        CalculateForces = Yes
        # PrintForces = Yes
    }}

    Options = {{
        WriteResultsTag = Yes
    }}

    ParserOptions = {{
        ParserVersion = 12
    }}"""

        with open(output_file, "w") as f:
            f.write(hsd_content)

    def _convert_to_results_tag(self, calc_dir):
        """Convert DFTB+ output to results.tag format for Phonopy."""
        detailed_out = calc_dir / "detailed.out"
        results_tag = calc_dir / "results.tag"

        if not detailed_out.exists():
            raise FileNotFoundError(f"detailed.out not found in {calc_dir}")

        # Extract forces from detailed.out
        _, forces, _ = self.extract_results(detailed_out)

        # Write results.tag in format expected by Phonopy
        with open(results_tag, "w") as f:
            f.write("forces [eV/AA]:\n")
            for force in forces:
                f.write(
                    f"  {force[0]:15.8f} {force[1]:15.8f} {force[2]:15.8f}\n"
                )

    def _calculate_phonon_properties(self, work_path, atoms):
        """Calculate phonon band structure, DOS, and thermodynamic properties."""
        try:
            import phonopy
            from phonopy import Phonopy
            from phonopy.file_IO import parse_FORCE_SETS
        except ImportError:
            raise ImportError(
                "Phonopy is required for phonon calculations. Install with: pip install phonopy"
            )

        results = {}

        # Load phonopy calculation
        try:
            # Try to load from phonopy_disp.yaml
            ph = phonopy.load(
                work_path / "phonopy_disp.yaml",
                force_sets_filename=work_path / "FORCE_SETS",
            )
        except:
            # Alternative: create Phonopy object manually
            from jarvis.core.atoms import ase_to_atoms
            from phonopy.structure.atoms import PhonopyAtoms

            jarvis_atoms = ase_to_atoms(atoms)
            unitcell = PhonopyAtoms(
                symbols=atoms.get_chemical_symbols(),
                scaled_positions=atoms.get_scaled_positions(),
                cell=atoms.get_cell(),
            )

            # Read supercell dimensions from displacement files
            disp_files = list(work_path.glob("geo.genS-*"))
            if disp_files:
                # Extract supercell info from first displacement file
                supercell_matrix = np.eye(3) * 2  # Default [2,2,2]

            ph = Phonopy(unitcell, supercell_matrix)

            # Load force sets
            if (work_path / "FORCE_SETS").exists():
                force_sets = parse_FORCE_SETS(work_path / "FORCE_SETS")
                ph.set_force_sets(force_sets)

        # Produce force constants
        ph.produce_force_constants()

        # Calculate phonon band structure
        band_paths = self._get_band_paths(atoms)
        if band_paths:
            ph.run_band_structure(band_paths, is_eigenvectors=True)
            bs_dict = ph.get_band_structure_dict()

            # Create phonon band structure plot
            self._plot_phonon_bands(bs_dict, work_path / "phonon_bands.png")

            results["band_structure"] = {
                "qpoints": bs_dict["qpoints"].tolist(),
                "frequencies": bs_dict["frequencies"].tolist(),
                "distances": bs_dict["distances"].tolist(),
                "labels": bs_dict.get("labels", []),
            }

        # Calculate phonon DOS
        ph.run_mesh(
            [20, 20, 20], with_eigenvectors=False, is_mesh_symmetry=False
        )
        ph.run_total_dos()
        dos_dict = ph.get_total_dos_dict()

        # Create phonon DOS plot
        self._plot_phonon_dos(dos_dict, work_path / "phonon_dos.png")

        results["dos"] = {
            "frequency_points": dos_dict["frequency_points"].tolist(),
            "total_dos": dos_dict["total_dos"].tolist(),
        }

        # Calculate thermodynamic properties
        ph.run_thermal_properties(t_step=10, t_max=1000)
        tp_dict = ph.get_thermal_properties_dict()

        results["thermal_properties"] = {
            "temperatures": tp_dict["temperatures"].tolist(),
            "free_energy": tp_dict["free_energy"].tolist(),
            "entropy": tp_dict["entropy"].tolist(),
            "heat_capacity": tp_dict["heat_capacity"].tolist(),
        }

        # Plot thermal properties
        self._plot_thermal_properties(
            tp_dict, work_path / "thermal_properties.png"
        )

        # Basic phonon analysis
        frequencies = ph.get_mesh_dict()["frequencies"]

        # Check for imaginary frequencies (instabilities)
        min_freq = np.min(frequencies)
        max_freq = np.max(frequencies)
        imaginary_modes = np.sum(
            frequencies < -1e-3
        )  # Tolerance for numerical noise

        results["analysis"] = {
            "min_frequency_THz": float(min_freq),
            "max_frequency_THz": float(max_freq),
            "imaginary_modes": int(imaginary_modes),
            "is_stable": bool(min_freq > -1e-3),
            "zero_point_energy_eV": float(
                np.sum(frequencies[frequencies > 0]) * 0.5 * 4.136e-3
            ),  # THz to eV
        }

        print(f"Phonon analysis:")
        print(f"  Minimum frequency: {min_freq:.3f} THz")
        print(f"  Imaginary modes: {imaginary_modes}")
        print(f"  Structure is {'stable' if min_freq > -1e-3 else 'unstable'}")

        return results

    def _get_band_paths(self, atoms):
        """Get high-symmetry k-point paths for phonon band structure."""
        try:
            from phonopy.structure.symmetry import symmetry
            from phonopy.structure.atoms import PhonopyAtoms

            unitcell = PhonopyAtoms(
                symbols=atoms.get_chemical_symbols(),
                scaled_positions=atoms.get_scaled_positions(),
                cell=atoms.get_cell(),
            )

            # Get high-symmetry points (simplified approach)
            # For a complete implementation, you'd want to use spglib/phonopy's
            # automatic k-path generation

            # Simple cubic path as example
            band_paths = [
                [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.5],
                ]
            ]

            return band_paths
        except:
            # Return None if automatic path generation fails
            return None

    def _plot_phonon_bands(self, bs_dict, output_file):
        """Create phonon band structure plot."""
        fig, ax = plt.subplots(figsize=(10, 6))

        distances = bs_dict["distances"]
        frequencies = bs_dict["frequencies"]

        # Plot each band
        for i in range(frequencies.shape[1]):
            ax.plot(
                distances,
                frequencies[:, i],
                "b-",
                linewidth=1.0,
                alpha=0.8,
            )

        ax.set_xlabel("Wave Vector")
        ax.set_ylabel("Frequency (THz)")
        ax.set_title("Phonon Band Structure")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_phonon_dos(self, dos_dict, output_file):
        """Create phonon DOS plot."""
        fig, ax = plt.subplots(figsize=(8, 6))

        frequencies = dos_dict["frequency_points"]
        dos = dos_dict["total_dos"]

        ax.plot(frequencies, dos, "k-", linewidth=1.5)
        ax.fill_between(frequencies, dos, alpha=0.3)

        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel("DOS")
        ax.set_title("Phonon Density of States")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_thermal_properties(self, tp_dict, output_file):
        """Create thermal properties plot."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        T = tp_dict["temperatures"]
        F = tp_dict["free_energy"]
        S = tp_dict["entropy"]
        Cv = tp_dict["heat_capacity"]

        # Free energy
        ax1.plot(T, F, "b-", linewidth=2)
        ax1.set_xlabel("Temperature (K)")
        ax1.set_ylabel("Free Energy (kJ/mol)")
        ax1.set_title("Helmholtz Free Energy")
        ax1.grid(True, alpha=0.3)

        # Entropy
        ax2.plot(T, S, "r-", linewidth=2)
        ax2.set_xlabel("Temperature (K)")
        ax2.set_ylabel("Entropy (J/K/mol)")
        ax2.set_title("Vibrational Entropy")
        ax2.grid(True, alpha=0.3)

        # Heat capacity
        ax3.plot(T, Cv, "g-", linewidth=2)
        ax3.set_xlabel("Temperature (K)")
        ax3.set_ylabel("Heat Capacity (J/K/mol)")
        ax3.set_title("Heat Capacity at Constant Volume")
        ax3.grid(True, alpha=0.3)

        # Combined plot
        ax4_twin = ax4.twinx()
        l1 = ax4.plot(T, S, "r-", label="Entropy", linewidth=2)
        l2 = ax4_twin.plot(T, Cv, "g-", label="Heat Capacity", linewidth=2)

        ax4.set_xlabel("Temperature (K)")
        ax4.set_ylabel("Entropy (J/K/mol)", color="r")
        ax4_twin.set_ylabel("Heat Capacity (J/K/mol)", color="g")
        ax4.tick_params(axis="y", labelcolor="r")
        ax4_twin.tick_params(axis="y", labelcolor="g")

        # Combine legends
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc="best")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def _json_serializable(obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def run_eos_bulkmod(
        self,
        atoms,
        work_dir="eos",
        volume_range=0.15,
        n_points=9,
        eos_type="birch_murnaghan",
        optimize_each=True,
    ):
        """
        Calculate equation of state and bulk modulus by volume deformation.

        Parameters:
        -----------
        atoms : ASE Atoms object
            Optimized atomic structure
        work_dir : str
            Working directory for EOS calculations
        volume_range : float
            Fractional volume range for deformation (±volume_range)
        n_points : int
            Number of volume points to calculate
        eos_type : str
            EOS model to fit ('birch_murnaghan', 'murnaghan', 'vinet', 'pourier_tarantola')
        optimize_each : bool
            Whether to optimize atomic positions at each volume

        Returns:
        --------
        dict : EOS results including bulk modulus, equilibrium volume, and fitted parameters
        """
        work_path = Path(work_dir)
        work_path.mkdir(exist_ok=True)

        print(f"Starting EOS calculation with {n_points} volume points...")
        print(f"Volume range: ±{volume_range*100:.1f}%")

        # Get initial volume and setup volume points
        V0 = atoms.get_volume()
        volume_factors = np.linspace(
            1.0 - volume_range, 1.0 + volume_range, n_points
        )
        volumes = V0 * volume_factors

        print(f"Initial volume: {V0:.3f} Å³")
        print(f"Volume points: {volumes[0]:.3f} to {volumes[-1]:.3f} Å³")

        # Calculate energy for each volume
        energies = []
        volume_dirs = []

        for i, (vol_factor, volume) in enumerate(zip(volume_factors, volumes)):
            vol_dir = work_path / f"vol_{i:02d}_{vol_factor:.3f}"
            vol_dir.mkdir(exist_ok=True)
            volume_dirs.append(vol_dir)

            print(
                f"Calculating volume point {i+1}/{n_points}: V/V0 = {vol_factor:.3f}"
            )

            # Scale cell uniformly to target volume
            scaled_atoms = atoms.copy()
            cell_scale = vol_factor ** (1 / 3)
            scaled_atoms.set_cell(
                atoms.get_cell() * cell_scale, scale_atoms=True
            )

            # Write geometry
            geo_file = vol_dir / "geo.gen"
            write(geo_file, scaled_atoms, format="gen")

            # Create input file
            if optimize_each:
                # Optimize positions but keep cell fixed
                self.create_eos_input(
                    scaled_atoms,
                    vol_dir / "dftb_in.hsd",
                    optimize_positions=True,
                    fix_cell=True,
                )
            else:
                # Single point calculation
                self.create_eos_input(
                    scaled_atoms,
                    vol_dir / "dftb_in.hsd",
                    optimize_positions=False,
                )

            # Run calculation
            try:
                self.run_command([self.dftb_executable], cwd=vol_dir)

                # Extract energy
                if optimize_each and (vol_dir / "geom.out.gen").exists():
                    # Read final geometry if optimization was performed
                    final_atoms = read(vol_dir / "geom.out.gen")
                    actual_volume = final_atoms.get_volume()
                else:
                    actual_volume = volume

                energy, _, _ = self.extract_results(vol_dir / "detailed.out")
                energies.append(energy)

                print(
                    f"  Energy: {energy:.6f} eV, Volume: {actual_volume:.3f} Å³"
                )

            except Exception as e:
                print(f"  Error in calculation: {e}")
                energies.append(None)

        # Filter out failed calculations
        valid_data = [
            (v, e) for v, e in zip(volumes, energies) if e is not None
        ]
        if len(valid_data) < 5:
            raise RuntimeError(
                f"Too few successful calculations ({len(valid_data)}). Need at least 5 points."
            )

        volumes_valid, energies_valid = zip(*valid_data)
        volumes_valid = np.array(volumes_valid)
        energies_valid = np.array(energies_valid)

        print(f"Successfully calculated {len(valid_data)} points")

        # Fit equation of state
        eos_results = self._fit_equation_of_state(
            volumes_valid, energies_valid, eos_type
        )

        # Calculate additional properties
        eos_results.update(
            self._calculate_eos_properties(
                volumes_valid, energies_valid, eos_results
            )
        )

        # Create plots
        self._plot_eos_fit(
            volumes_valid,
            energies_valid,
            eos_results,
            work_path / "eos_fit.png",
            eos_type,
        )

        # Save detailed results
        results = {
            "initial_volume_A3": float(V0),
            "volume_range": volume_range,
            "n_points_calculated": len(valid_data),
            "n_points_requested": n_points,
            "eos_type": eos_type,
            "optimize_each_volume": optimize_each,
            "raw_data": {
                "volumes_A3": volumes_valid.tolist(),
                "energies_eV": energies_valid.tolist(),
            },
            "eos_fit": eos_results,
            "bulk_modulus_GPa": eos_results.get("bulk_modulus_GPa"),
            "equilibrium_volume_A3": eos_results.get("V0_fit"),
            "cohesive_energy_eV": eos_results.get("E0_fit"),
        }

        # Save results
        with open(work_path / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=self._json_serializable)

        # Print summary
        print("\n=== EOS Results Summary ===")
        print(f"EOS Model: {eos_type}")
        print(
            f"Bulk Modulus: {eos_results.get('bulk_modulus_GPa', 'N/A'):.1f} GPa"
        )
        print(f"Equilibrium Volume: {eos_results.get('V0_fit', 'N/A'):.3f} Å³")
        print(f"Cohesive Energy: {eos_results.get('E0_fit', 'N/A'):.6f} eV")
        if "B0_prime" in eos_results:
            print(f"B₀': {eos_results['B0_prime']:.2f}")

        return results

    # def run_eos_bulkmod(self, atoms, work_dir="eos", volume_range=0.15, n_points=9,
    def create_eos_input(
        self, atoms, output_file, optimize_positions=False, fix_cell=False
    ):
        """Create DFTB+ input file for EOS calculations."""
        maxl_block = self.get_max_angular_momentum(atoms)

        hsd_content = f"""Geometry = GenFormat {{
        <<< "geo.gen"
    }}

    Hamiltonian = DFTB {{
        SCC = Yes
        MaxAngularMomentum = {{
    {maxl_block}
        }}
        SlaterKosterFiles = Type2FileNames {{
            Prefix = "{self.sk_dir}/"
            Separator = "-"
            Suffix = ".skf"
        }}
        KPointsAndWeights = SuperCellFolding {{
            {self.k_mesh[0]} 0 0
            0 {self.k_mesh[1]} 0
            0 0 {self.k_mesh[2]}
            0.0 0.0 0.0
        }}
        Filling = Fermi {{
            Temperature [Kelvin] = 0
        }}
    }}

    Analysis = {{
        CalculateForces = Yes
    }}

    Options = {{
        WriteResultsTag = Yes
    }}

    ParserOptions = {{
        ParserVersion = 12
    }}"""

        # Add geometry optimization if requested
        if optimize_positions:
            if fix_cell:
                driver_block = """
    Driver = GeometryOptimization {
        Optimizer = Rational {}
        MovedAtoms = 1:-1
        MaxSteps = 50
        OutputPrefix = "geom.out"
        Convergence = {
            GradElem = 1E-4
        }
        LatticeOpt = No
    }"""
            else:
                driver_block = """
    Driver = GeometryOptimization {
        Optimizer = Rational {}
        MovedAtoms = 1:-1
        MaxSteps = 50
        OutputPrefix = "geom.out"
        Convergence = {
            GradElem = 1E-4
        }
        LatticeOpt = Yes
    }"""
            hsd_content = hsd_content.replace(
                "Geometry =", driver_block + "\n\nGeometry ="
            )

        with open(output_file, "w") as f:
            f.write(hsd_content)

    # def create_eos_input(self, atoms, output_file, optimize_positions=False, fix_cell=False):
    def _fit_equation_of_state(
        self, volumes, energies, eos_type="birch_murnaghan"
    ):
        """Fit equation of state to volume-energy data."""
        from scipy.optimize import minimize

        # Convert to per-atom quantities
        n_atoms = len(volumes)  # This is wrong, should be number of atoms
        # We need to get number of atoms somehow - let's assume it's stored
        # For now, we'll work with total energies and volumes

        def birch_murnaghan(V, E0, V0, B0, B0_prime):
            """Birch-Murnaghan equation of state."""
            eta = (V0 / V) ** (2 / 3)
            return E0 + (9 * V0 * B0 / 16) * (
                (eta - 1) ** 3 * B0_prime + (eta - 1) ** 2 * (6 - 4 * eta)
            )

        def murnaghan(V, E0, V0, B0, B0_prime):
            """Murnaghan equation of state."""
            return (
                E0
                + B0
                * V
                / B0_prime
                * (((V0 / V) ** B0_prime) / (B0_prime - 1) + 1)
                - V0 * B0 / (B0_prime - 1)
            )

        def vinet(V, E0, V0, B0, B0_prime):
            """Vinet equation of state."""
            eta = (V / V0) ** (1 / 3)
            xi = 1.5 * (B0_prime - 1)
            return E0 + 2 * B0 * V0 / (B0_prime - 1) ** 2 * (
                2
                - (5 + 3 * xi * (eta - 1) - 3 * eta) * np.exp(-xi * (eta - 1))
            )

        def pourier_tarantola(V, E0, V0, B0, B0_prime):
            """Pourier-Tarantola equation of state."""
            eta = (V / V0) ** (1 / 3)
            return E0 + B0 * V0 / B0_prime * (
                ((eta) ** B0_prime - 1) / B0_prime + eta - 1
            )

        # Select EOS function
        eos_functions = {
            "birch_murnaghan": birch_murnaghan,
            "murnaghan": murnaghan,
            "vinet": vinet,
            "pourier_tarantola": pourier_tarantola,
        }

        if eos_type not in eos_functions:
            raise ValueError(
                f"Unknown EOS type: {eos_type}. Available: {list(eos_functions.keys())}"
            )

        eos_func = eos_functions[eos_type]

        # Initial parameter guess
        E0_guess = np.min(energies)
        V0_guess = volumes[np.argmin(energies)]

        # Estimate bulk modulus from curvature
        # B0 = V * d²E/dV² (at equilibrium)
        # Rough estimate using finite differences
        dV = np.mean(np.diff(volumes))
        d2E_dV2 = np.gradient(np.gradient(energies, dV), dV)
        B0_guess = (
            abs(V0_guess * d2E_dV2[np.argmin(energies)]) * 160.2176
        )  # eV/Å³ to GPa
        B0_guess = max(B0_guess, 50)  # Minimum reasonable value

        B0_prime_guess = 4.0  # Typical value

        initial_guess = [E0_guess, V0_guess, B0_guess, B0_prime_guess]

        # Define objective function
        def objective(params):
            try:
                E_calc = eos_func(volumes, *params)
                return np.sum((energies - E_calc) ** 2)
            except:
                return 1e10

        # Set bounds for parameters
        bounds = [
            (None, None),  # E0
            (min(volumes) * 0.8, max(volumes) * 1.2),  # V0
            (1, 1000),  # B0 (GPa)
            (1, 10),  # B0'
        ]

        # Fit the equation of state
        try:
            result = minimize(
                objective, initial_guess, bounds=bounds, method="L-BFGS-B"
            )
            if result.success:
                E0_fit, V0_fit, B0_fit, B0_prime_fit = result.x
                fitted_energies = eos_func(volumes, *result.x)
                r_squared = 1 - np.sum(
                    (energies - fitted_energies) ** 2
                ) / np.sum((energies - np.mean(energies)) ** 2)

                return {
                    "E0_fit": float(E0_fit),
                    "V0_fit": float(V0_fit),
                    "bulk_modulus_GPa": float(B0_fit),
                    "B0_prime": float(B0_prime_fit),
                    "r_squared": float(r_squared),
                    "fitted_energies": fitted_energies.tolist(),
                    "fit_success": True,
                    "fit_message": "Successful fit",
                }
            else:
                return {
                    "fit_success": False,
                    "fit_message": f"Optimization failed: {result.message}",
                }
        except Exception as e:
            return {
                "fit_success": False,
                "fit_message": f"Fitting error: {str(e)}",
            }

    def _calculate_eos_properties(self, volumes, energies, eos_results):
        """Calculate additional properties from EOS fit."""
        if not eos_results.get("fit_success", False):
            return {}

        properties = {}

        # Compressibility
        B0 = eos_results["bulk_modulus_GPa"]
        properties["compressibility_1_GPa"] = 1.0 / B0

        # Volume derivative of bulk modulus
        if "B0_prime" in eos_results:
            properties["bulk_modulus_derivative"] = eos_results["B0_prime"]

        # Fit quality metrics
        if "fitted_energies" in eos_results:
            fitted_energies = np.array(eos_results["fitted_energies"])
            residuals = energies - fitted_energies
            properties["rms_error_eV"] = float(np.sqrt(np.mean(residuals**2)))
            properties["max_error_eV"] = float(np.max(np.abs(residuals)))
            properties["mean_abs_error_eV"] = float(np.mean(np.abs(residuals)))

        # Pressure at each volume (for checking)
        # P = -dE/dV
        if len(volumes) > 2:
            pressures = (
                -np.gradient(energies, volumes) * 160.2176
            )  # eV/Å³ to GPa
            properties["pressures_GPa"] = pressures.tolist()

        return properties

    def _plot_eos_fit(
        self, volumes, energies, eos_results, output_file, eos_type
    ):
        """Create EOS fit plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Main EOS plot
        ax1.scatter(
            volumes, energies, color="red", s=50, zorder=5, label="DFTB+ data"
        )

        if (
            eos_results.get("fit_success", False)
            and "fitted_energies" in eos_results
        ):
            # Plot fitted curve
            V_fine = np.linspace(min(volumes), max(volumes), 100)

            # Get EOS function for fine grid
            if eos_type == "birch_murnaghan":

                def eos_func(V, E0, V0, B0, B0_prime):
                    eta = (V0 / V) ** (2 / 3)
                    return E0 + (9 * V0 * B0 / 16) * (
                        (eta - 1) ** 3 * B0_prime
                        + (eta - 1) ** 2 * (6 - 4 * eta)
                    )

            elif eos_type == "murnaghan":

                def eos_func(V, E0, V0, B0, B0_prime):
                    return (
                        E0
                        + B0
                        * V
                        / B0_prime
                        * (((V0 / V) ** B0_prime) / (B0_prime - 1) + 1)
                        - V0 * B0 / (B0_prime - 1)
                    )

            elif eos_type == "vinet":

                def eos_func(V, E0, V0, B0, B0_prime):
                    eta = (V / V0) ** (1 / 3)
                    xi = 1.5 * (B0_prime - 1)
                    return E0 + 2 * B0 * V0 / (B0_prime - 1) ** 2 * (
                        2
                        - (5 + 3 * xi * (eta - 1) - 3 * eta)
                        * np.exp(-xi * (eta - 1))
                    )

            else:  # pourier_tarantola

                def eos_func(V, E0, V0, B0, B0_prime):
                    eta = (V / V0) ** (1 / 3)
                    return E0 + B0 * V0 / B0_prime * (
                        ((eta) ** B0_prime - 1) / B0_prime + eta - 1
                    )

            try:
                E_fine = eos_func(
                    V_fine,
                    eos_results["E0_fit"],
                    eos_results["V0_fit"],
                    eos_results["bulk_modulus_GPa"],
                    eos_results["B0_prime"],
                )
                ax1.plot(
                    V_fine,
                    E_fine,
                    "b-",
                    linewidth=2,
                    label=f'{eos_type.replace("_", "-").title()} fit',
                )

                # Mark equilibrium
                ax1.axvline(
                    eos_results["V0_fit"],
                    color="green",
                    linestyle="--",
                    alpha=0.7,
                    label=f'V₀ = {eos_results["V0_fit"]:.3f} Å³',
                )
            except:
                pass

        ax1.set_xlabel("Volume (Å³)")
        ax1.set_ylabel("Energy (eV)")
        ax1.set_title("Equation of State")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Residuals plot
        if (
            eos_results.get("fit_success", False)
            and "fitted_energies" in eos_results
        ):
            fitted_energies = np.array(eos_results["fitted_energies"])
            residuals = (energies - fitted_energies) * 1000  # Convert to meV

            ax2.scatter(volumes, residuals, color="red", s=50)
            ax2.axhline(0, color="black", linestyle="-", alpha=0.5)
            ax2.set_xlabel("Volume (Å³)")
            ax2.set_ylabel("Residuals (meV)")
            ax2.set_title(
                f'Fit Residuals (R² = {eos_results.get("r_squared", 0):.4f})'
            )
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(
                0.5,
                0.5,
                "Fit Failed",
                transform=ax2.transAxes,
                ha="center",
                va="center",
                fontsize=16,
            )
            ax2.set_title("Fit Status")

        # Add text box with results
        if eos_results.get("fit_success", False):
            textstr = f"""B₀ = {eos_results["bulk_modulus_GPa"]:.1f} GPa
    B₀' = {eos_results["B0_prime"]:.2f}
    V₀ = {eos_results["V0_fit"]:.3f} Å³"""
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
            ax1.text(
                0.02,
                0.98,
                textstr,
                transform=ax1.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=props,
            )

            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"EOS plot saved to {output_file}")

    def compare_bulk_modulus(self, experimental_B0=None, work_dir="eos"):
        """Compare calculated bulk modulus with experimental value."""
        results_file = Path(work_dir) / "eos_results.json"

        if not results_file.exists():
            print("No EOS results found. Run run_eos_bulkmod() first.")
            return

        with open(results_file, "r") as f:
            results = json.load(f)

        calc_B0 = results.get("bulk_modulus_GPa")

        if calc_B0 is None:
            print("No bulk modulus found in results.")
            return

        print(f"\n=== Bulk Modulus Comparison ===")
        print(f"Calculated (DFTB+): {calc_B0:.1f} GPa")

        if experimental_B0 is not None:
            print(f"Experimental:       {experimental_B0:.1f} GPa")
            error = ((calc_B0 - experimental_B0) / experimental_B0) * 100
            print(f"Relative Error:     {error:+.1f}%")

            if abs(error) < 10:
                print("✓ Good agreement with experiment")
            elif abs(error) < 25:
                print("⚠ Moderate agreement with experiment")
            else:
                print("✗ Poor agreement with experiment")

        return {
            "calculated_GPa": calc_B0,
            "experimental_GPa": experimental_B0,
            "relative_error_percent": error if experimental_B0 else None,
        }


def download_vasp_data(jid):
    """Download VASP data from JARVIS database."""
    dat = get_jid_data(jid=jid, dataset="dft_3d")
    atoms = Atoms.from_dict(dat["atoms"])
    # atoms = atoms.get_conventional_atoms
    atoms = atoms.ase_converter()

    # Find band structure calculation
    for raw_file in dat["raw_files"]:
        if "Bandst" in raw_file:
            calc_zipfile_link = raw_file.split(",")[2]
            r = requests.get(calc_zipfile_link)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            vrun_content = z.read("vasprun.xml").decode("utf-8")

            # Create temporary file
            fd, path = tempfile.mkstemp()
            with os.fdopen(fd, "w") as tmp:
                tmp.write(vrun_content)

            vrun = Vasprun(path)
            kpoints = extract_kpoints_from_vasprun(path)
            return atoms, vrun, kpoints

    raise ValueError("No band structure data found")


def main(
    jid="JVASP-816",
    atoms=None,
    dftb_executable="dftb+",
    sk_dir="ParameterSets/ptbp/complete_set",
    k_mesh=[10, 10, 10],
    config=None,
):
    """Main execution function."""
    cwd = os.getcwd()
    work_path = Path(jid + "_dftb")
    work_path.mkdir(exist_ok=True)
    os.chdir(work_path)
    # TODO: Run specific jobs as per config
    # TODO: pass only config to main
    try:

        # Configuration

        # Check if paths exist
        if not os.path.exists(dftb_executable):
            print(f"Error: DFTB+ executable not found at {dftb_executable}")
            print("Please check the path or install DFTB+")
            return

        if not os.path.exists(sk_dir):
            print(
                f"Error: Slater-Koster files directory not found at {sk_dir}"
            )
            print("Please check the path or download the SK files")
            return

        # Download data
        print(f"Downloading data for {jid}...")
        try:
            atoms, vasprun, kpoints = download_vasp_data(jid)
        except Exception as e:
            print(f"Error downloading VASP data: {e}")
            return

        # Initialize calculator
        try:
            calc = DFTBCalculator(dftb_executable, sk_dir, k_mesh)
        except FileNotFoundError as e:
            print(f"Error initializing calculator: {e}")
            return

        # Run optimization
        print("Running geometry optimization...")
        try:
            opt_results, final_atoms = calc.run_optimization(atoms)
            print(f"Final energy: {opt_results['energy']:.4f} eV")
            print(f"Fermi level: {opt_results['fermi_level']:.4f} eV")
            print(
                f"Electronic properties saved: opt/electronic_properties.json"
            )
            print(f"DOS plot saved: opt/dos.png")
        except Exception as e:
            print(f"Error during optimization: {e}")
            return

        print("Running EOS...")
        try:
            eos_results = calc.run_eos_bulkmod(
                final_atoms, work_dir="eos", volume_range=0.15, n_points=9
            )
        except Exception as e:
            print(f"Error during optimization: {e}")
            return

        try:
            # Run phonon calculation
            phonon_results = calc.run_phonon(
                final_atoms,
                work_dir="phonon",
                supercell=[2, 2, 2],
                amplitude=5e-4,
                gamma_centered=True,
            )
        except Exception as e:
            print(f"Error during optimization: {e}")
            return
        # Run band structure
        print("Running band structure calculation...")
        try:
            band_results = calc.run_band_structure(
                final_atoms, kpoints, vasprun
            )
            print(f"Bandgap: {band_results['bandgap']:.4f} eV")
            print(
                f"VBM: {band_results['vbm']:.4f} eV"
                if band_results["vbm"]
                else "VBM: Metallic"
            )
            print(
                f"CBM: {band_results['cbm']:.4f} eV"
                if band_results["cbm"]
                else "CBM: Metallic"
            )
            if band_results["max_difference"]:
                print(
                    f"Maximum band difference vs VASP: {band_results['max_difference']:.4f} eV"
                )

            # Print file locations
            print(
                f"Electronic properties saved: band/electronic_properties.json"
            )
            print(f"Band structure plot saved: band/band_structure.png")
            print(f"DOS plot saved: band/dos.png")

            # Print summary
            print("\n=== Calculation Summary ===")
            print(f"System: {jid}")
            print(f"Total energy: {band_results['energy']:.4f} eV")
            print(f"Fermi level: {band_results['fermi_level']:.4f} eV")
            print(f"Bandgap: {band_results['bandgap']:.4f} eV")

        except Exception as e:
            print(f"Error during band structure calculation: {e}")
            return

        print("Calculations completed successfully!")
        print("\nOutput structure:")
        print("opt/")
        print("  ├── electronic_properties.json")
        print("  ├── dos.png")
        print("  └── results.json")
        print("band/")
        print("  ├── electronic_properties.json")
        print("  ├── band_structure.png")
        print("  ├── dos.png")
        print("  └── band_results.json")
    except:
        pass

    os.chdir(cwd)


if __name__ == "__main__":
    main()
