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
        Prefix = "{self.sk_dir.rstrip('/')}/"
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

        # Run calculation
        self.run_command([self.dftb_executable], cwd=work_path)

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
        bandgap, vbm, cbm = self.calculate_band_properties(
            band_energies_adjusted
        )

        # Save electronic properties JSON
        electronic_props = self.save_electronic_properties(
            work_path, energy, fermi_ev
        )

        results = {
            "energy": energy,
            "forces": forces.tolist(),
            "fermi_level": fermi_ev,
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
        bandgap, vbm, cbm = self.calculate_band_properties(
            band_energies_adjusted
        )

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


def download_vasp_data(jid):
    """Download VASP data from JARVIS database."""
    dat = get_jid_data(jid=jid, dataset="dft_3d")
    atoms = Atoms.from_dict(dat["atoms"]).ase_converter()

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
    jid="JVASP-1002",
    atoms=None,
    dftb_executable="/wrk/knc6/Software/dftb/bin/dftb+",
    sk_dir="/wrk/knc6/chipstb/chipstb/siband-1.1.0/skfiles/",
    k_mesh=[10, 10, 10],
):
    """Main execution function."""
    cwd = os.getcwd()
    name = jid + "_dftb"
    work_path = Path(name)
    work_path.mkdir(exist_ok=True)
    os.chdir(work_path)
    try:

        # Configuration

        # sk_dir = "/content/matsci-0.3.0/skfiles/"

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
