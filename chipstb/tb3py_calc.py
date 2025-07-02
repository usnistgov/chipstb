import os
import json
import shutil
import subprocess
import tempfile
import zipfile
import io
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from chipstb.utils import (
    extract_kpoints_from_vasprun,
    download_jarvis_dft_data,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import requests
import re
import time
from ase.io import read, write
from jarvis.core.atoms import Atoms, ase_to_atoms
from jarvis.db.figshare import get_jid_data
from jarvis.io.vasp.outputs import Vasprun


class TB3PyCalc:
    """
    Comprehensive TB3Py (ThreeBodyTB) calculator for geometry optimization,
    band structure, phonons, and bulk modulus calculations.
    Organized similar to DFTBCalculator with folder structure and JSON output.
    """

    def __init__(self, julia_executable="julia", work_dir="."):
        """
        Initialize TB3PyCalc calculator

        Args:
            julia_executable: Path to Julia executable
            work_dir: Base working directory for calculations
        """
        self.julia_executable = julia_executable
        self.work_dir = Path(work_dir)
        self.energy = None
        self.tbc = None
        self.results = {}

        # Check if Julia executable exists
        if not shutil.which(julia_executable):
            raise FileNotFoundError(
                f"Julia executable not found: {julia_executable}"
            )

        # Ensure work directory exists
        self.work_dir.mkdir(exist_ok=True)

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

    def write_poscar(
        self, structure: str, filename: str = "POSCAR", work_dir=None
    ):
        """Write POSCAR file from structure string"""
        if work_dir is None:
            work_dir = self.work_dir
        poscar_path = Path(work_dir) / filename
        with open(poscar_path, "w") as f:
            f.write(structure)

    def _run_julia_script(
        self, script_content: str, script_name: str = "input.jl", work_dir=None
    ) -> subprocess.CompletedProcess:
        """Execute Julia script and return result"""
        if work_dir is None:
            work_dir = self.work_dir

        script_path = Path(work_dir) / script_name

        with open(script_path, "w") as f:
            f.write(script_content)

        # Run Julia command
        self.run_command([self.julia_executable, script_name], cwd=work_dir)

    def _read_output_files(self, filenames: List[str], work_dir=None) -> Dict:
        """Read output files and return as dictionary"""
        if work_dir is None:
            work_dir = self.work_dir

        results = {}
        for filename in filenames:
            filepath = Path(work_dir) / filename
            if filepath.exists():
                with open(filepath, "r") as f:
                    content = f.read().strip()
                    try:
                        # Try to convert to float if possible
                        results[filename] = float(content)
                    except ValueError:
                        results[filename] = content
            else:
                results[filename] = None
        return results

    def calculate_band_properties(self, band_energies, fermi_level=0.0):
        """Calculate band properties: bandgap, VBM, CBM."""
        # Adjust bands relative to Fermi level
        band_energies_adj = band_energies - fermi_level

        # Find VBM and CBM
        vbm = -float("inf")
        cbm = float("inf")

        for kpt_bands in band_energies_adj:
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

    def run_optimization(self, atoms, work_dir="opt"):
        """
        Run geometry optimization calculation.

        Args:
            atoms: ASE Atoms object
            work_dir: Working directory for optimization

        Returns:
            Dictionary with optimization results and final structure
        """
        t1 = time.time()
        work_path = Path(work_dir)
        work_path.mkdir(exist_ok=True)

        # Write initial structure
        initial_poscar = self._atoms_to_poscar(atoms)
        self.write_poscar(initial_poscar, "POSCAR", work_path)

        script = """
using ThreeBodyTB
using NPZ

println("Starting structure relaxation...")
crys = makecrys("POSCAR")
cfinal, tbc, energy, force, stress = relax_structure(crys)

println("Final energy: ", energy)
println("Max force: ", maximum(abs.(force)))
println("Max stress: ", maximum(abs.(stress)))

# Save optimized structure
ThreeBodyTB.CrystalMod.write_poscar(cfinal, "POSCAR_relaxed")

# Save results
open("final_energy", "w") do file
    write(file, string(energy))
end

open("fermi_energy", "w") do file
    write(file, string(tbc.efermi))
end

open("max_force", "w") do file
    write(file, string(maximum(abs.(force))))
end

open("max_stress", "w") do file
    write(file, string(maximum(abs.(stress))))
end

npzwrite("final_forces.npz", force)
npzwrite("final_stress.npz", stress)

# Save tight-binding data
ThreeBodyTB.TB.write_tb_crys("tbc_relaxed.xml.gz", tbc)

# Get band summary
band_info = ThreeBodyTB.BandStruct.band_summary(tbc)
open("band_summary", "w") do file
    write(file, string(band_info))
end

# Get charge information
dq = ThreeBodyTB.TB.get_dq(tbc)
open("dq", "w") do file
    write(file, string(dq))
end

println("Structure relaxation completed.")
"""

        try:
            relaxed_poscar_path = work_path / "POSCAR_relaxed"
            if not os.path.exists(relaxed_poscar_path):
                self._run_julia_script(script, "relax.jl", work_path)

            # Read results
            output_data = self._read_output_files(
                [
                    "final_energy",
                    "fermi_energy",
                    "max_force",
                    "max_stress",
                    "band_summary",
                    "dq",
                ],
                work_path,
            )

            # Read optimized structure
            relaxed_poscar_path = work_path / "POSCAR_relaxed"
            final_atoms = atoms  # Default fallback
            optimized_structure = None

            if relaxed_poscar_path.exists():
                with open(relaxed_poscar_path, "r") as f:
                    optimized_structure = f.read()
                # Try to read as ASE atoms
                try:
                    final_atoms = read(relaxed_poscar_path, format="vasp")
                except:
                    pass

            # Parse band summary for gap information
            bandgap, vbm, cbm = 0.0, None, None
            if output_data.get("band_summary"):
                band_summary = str(output_data["band_summary"])
                # Try to extract gap info - this depends on TB3 output format
                # You may need to adjust this parsing based on actual output

            # Save electronic properties
            electronic_props = self.save_electronic_properties(
                work_path,
                output_data.get("final_energy"),
                output_data.get("fermi_energy"),
                bandgap,
                vbm,
                cbm,
            )

            t2 = time.time()
            # Compile results
            results = {
                "energy": output_data.get("final_energy"),
                "fermi_level": output_data.get("fermi_energy"),
                "max_force": output_data.get("max_force"),
                "max_stress": output_data.get("max_stress"),
                "bandgap": bandgap,
                "vbm": vbm,
                "cbm": cbm,
                "time": t2 - t1,
                "band_summary": output_data.get("band_summary"),
                "charge_transfer": output_data.get("dq"),
                "initial_atoms": ase_to_atoms(atoms).to_dict(),
                "final_atoms": ase_to_atoms(final_atoms).to_dict(),
                "optimized_structure": optimized_structure,
                "electronic_properties": electronic_props,
                "convergence": {
                    "force_converged": output_data.get(
                        "max_force", float("inf")
                    )
                    < 0.01,
                    "stress_converged": output_data.get(
                        "max_stress", float("inf")
                    )
                    < 0.01,
                },
            }

            # Save results
            with open(work_path / "results.json", "w") as f:
                json.dump(
                    results, f, indent=2, default=self._json_serializable
                )

            self.results["optimization"] = results
            print(f"Optimization completed. Results saved in {work_path}/")

            return results, final_atoms

        except Exception as e:
            print(f"Error during optimization: {e}")
            raise

    def run_band_structure(
        self, atoms, kpoints, vasprun=None, work_dir="band"
    ):
        """
        Run band structure calculation with comprehensive analysis.

        Args:
            atoms: ASE Atoms object (optimized structure)
            kpoints: List of k-points for band structure
            vasprun: Optional VASP vasprun object for comparison
            work_dir: Working directory for band calculation

        Returns:
            Dictionary with band structure results
        """
        t1 = time.time()
        work_path = Path(work_dir)
        work_path.mkdir(exist_ok=True)

        # Copy optimized structure and TB data from optimization
        opt_path = Path("opt")
        if opt_path.exists():
            if (opt_path / "POSCAR_relaxed").exists():
                shutil.copy(opt_path / "POSCAR_relaxed", work_path / "POSCAR")
            elif (opt_path / "POSCAR").exists():
                shutil.copy(opt_path / "POSCAR", work_path / "POSCAR")
            if (opt_path / "tbc_relaxed.xml.gz").exists():
                shutil.copy(
                    opt_path / "tbc_relaxed.xml.gz", work_path / "tbc.xml.gz"
                )
        else:
            # Write current structure
            poscar_content = self._atoms_to_poscar(atoms)
            self.write_poscar(poscar_content, "POSCAR", work_path)

        # Convert kpoints to TB3 format
        # kpath_str = self._format_kpoints_for_tb3(kpoints)
        kpath_str = kpoints
        script = f"""
using ThreeBodyTB
using NPZ
using Plots

println("Calculating band structure...")

# Load or calculate tight-binding data
if isfile("tbc.xml.gz")
    println("Loading existing TB data...")
    tbc = ThreeBodyTB.TB.read_tb_crys("tbc.xml.gz")
else
    println("Calculating TB data from scratch...")
    crys = makecrys("POSCAR")
    energy, tbc, flag = scf_energy(crys)
    ThreeBodyTB.TB.write_tb_crys("tbc.xml.gz", tbc)
end

write_hr_dat(tbc,filename="my_hr.dat")
vals = calc_bands(tbc,{kpoints})
#vals = calc_bands(tbc, [0.0 0.0 0.0; 0.0 0.0 0.1])
efermi=tbc.efermi
# Calculate band structure
#println("Computing bands along k-path...")
#bands_data = ThreeBodyTB.BandStruct.band_structure(tbc, kpath, nk)

# Extract data
#kpts = bands_data[1]
#bands = bands_data[2] 
#efermi = bands_data[3]

#println("Band structure calculated.")
#println("Number of k-points: ", size(kpts, 1))
#println("Number of bands: ", size(bands, 2))
#println("Fermi energy: ", efermi)

# Save results
#npzwrite("kpoints.npz", kpts)
npzwrite("bands.npz", vals)
open("fermi_level", "w") do file
    write(file, string(efermi))
end

# Get band summary
band_info = ThreeBodyTB.BandStruct.band_summary(tbc)
open("band_info", "w") do file
    write(file, string(band_info))
end

# Save total energy
#energy = tbc.scf_energy
#open("total_energy", "w") do file
#    write(file, string(energy))
#end
tbc_new, plot_object = plot_bandstr_dos(tbc);
savefig(plot_object, "bandstructure_dos.png")
println("Band structure calculation completed.")
"""

        try:
            self._run_julia_script(script, "bands.jl", work_path)

            # Read results
            output_data = self._read_output_files(
                ["fermi_level", "band_info", "total_energy"], work_path
            )

            # Load band structure data
            kpts_file = work_path / "kpoints.npz"
            bands_file = work_path / "bands.npz"

            if kpts_file.exists() and bands_file.exists():
                kpts_data = np.load(kpts_file)["arr_0"]
                bands_data = np.load(bands_file)["arr_0"]

                # Adjust bands relative to Fermi level
                fermi_level = output_data.get("fermi_level", 0.0)
                bands_adjusted = bands_data - fermi_level

                # Calculate band properties
                bandgap, vbm, cbm = self.calculate_band_properties(
                    bands_adjusted, 0.0
                )

                # Create band structure plot
                band_plot_data = self.create_band_structure_plot(
                    np.arange(len(kpts_data)),
                    bands_adjusted,
                    work_path / "band_structure.png",
                )

            else:
                kpts_data = []
                bands_adjusted = []
                bandgap, vbm, cbm = None, None, None
                band_plot_data = None

            # Save electronic properties
            electronic_props = self.save_electronic_properties(
                work_path,
                output_data.get("total_energy"),
                output_data.get("fermi_level"),
                bandgap,
                vbm,
                cbm,
            )

            # Compare with VASP if provided
            vasp_comparison = None
            if vasprun:
                try:
                    vasp_comparison = self._compare_with_vasp(
                        work_path, vasprun, bands_adjusted, fermi_level
                    )
                except Exception as e:
                    print(f"VASP comparison failed: {e}")

            t2 = time.time()
            # Compile results
            results = {
                "energy": output_data.get("total_energy"),
                "fermi_level": output_data.get("fermi_level"),
                "bandgap": bandgap,
                "vbm": vbm,
                "time": t2 - t1,
                "cbm": cbm,
                "band_info": output_data.get("band_info"),
                "electronic_properties": electronic_props,
                "band_structure": {
                    "kpoints": (
                        kpts_data.tolist() if len(kpts_data) > 0 else []
                    ),
                    "energies": (
                        bands_adjusted.tolist()
                        if len(bands_adjusted) > 0
                        else []
                    ),
                    "plot_data": band_plot_data,
                },
                "vasp_comparison": vasp_comparison,
            }

            # Save results
            with open(work_path / "results.json", "w") as f:
                json.dump(
                    results, f, indent=2, default=self._json_serializable
                )

            self.results["band_structure"] = results
            print(f"Band structure completed. Results saved in {work_path}/")

            return results

        except Exception as e:
            print(f"Error during band structure calculation: {e}")
            raise

    def run_eos_bulkmod(
        self, atoms, work_dir="eos", volume_range=0.15, n_points=9
    ):
        """
        Calculate equation of state and bulk modulus by volume deformation.

        Args:
            atoms: ASE Atoms object (optimized structure)
            work_dir: Working directory for EOS calculations
            volume_range: Fractional volume range for deformation
            n_points: Number of volume points to calculate

        Returns:
            Dictionary with EOS results including bulk modulus
        """
        t1 = time.time()
        work_path = Path(work_dir)
        work_path.mkdir(exist_ok=True)

        print(f"Starting EOS calculation with {n_points} volume points...")
        print(f"Volume range: ±{volume_range*100:.1f}%")

        # Get initial volume
        V0 = atoms.get_volume()
        volume_factors = np.linspace(
            1.0 - volume_range, 1.0 + volume_range, n_points
        )

        print(f"Initial volume: {V0:.3f} Å³")

        # Calculate energy for each volume
        energies = []
        volumes = []

        for i, vol_factor in enumerate(volume_factors):
            vol_dir = work_path / f"vol_{i:02d}_{vol_factor:.3f}"
            vol_dir.mkdir(exist_ok=True)

            print(
                f"Calculating volume point {i+1}/{n_points}: V/V0 = {vol_factor:.3f}"
            )

            # Scale cell uniformly
            scaled_atoms = atoms.copy()
            cell_scale = vol_factor ** (1 / 3)
            scaled_atoms.set_cell(
                atoms.get_cell() * cell_scale, scale_atoms=True
            )

            # Write scaled structure
            scaled_poscar = self._atoms_to_poscar(scaled_atoms)
            self.write_poscar(scaled_poscar, "POSCAR", vol_dir)

            # Create Julia script for single point calculation
            script = """
using ThreeBodyTB
using LinearAlgebra
println("Calculating energy for scaled structure...")
crys = makecrys("POSCAR")
energy, tbc, flag = scf_energy(crys)
A=crys.A;v1 = A[1, :];v2 = A[2, :];v3 = A[3, :];vol = abs(dot(cross(v1, v2), v3));
println("Energy: ", energy)
println("Volume: ", vol)

open("energy", "w") do file
    write(file, string(energy))
end

open("volume", "w") do file
    write(file, string(vol))
end

println("Calculation completed.")
"""

            try:
                self._run_julia_script(script, "eos_point.jl", vol_dir)

                # Read results
                vol_results = self._read_output_files(
                    ["energy", "volume"], vol_dir
                )
                energy = vol_results.get("energy")
                volume = vol_results.get("volume")

                if energy is not None and volume is not None:
                    energies.append(energy)
                    volumes.append(volume)
                    print(
                        f"  Energy: {energy:.6f} eV, Volume: {volume:.3f} Å³"
                    )
                else:
                    print(f"  Failed to read results")

            except Exception as e:
                print(f"  Error in calculation: {e}")

        # Fit equation of state
        if len(energies) < 5:
            raise RuntimeError(
                f"Too few successful calculations ({len(energies)}). Need at least 5 points."
            )

        volumes = np.array(volumes)
        energies = np.array(energies)

        # Simple quadratic fit for bulk modulus
        # E(V) = E0 + a*(V-V0) + b*(V-V0)^2
        # B = V * d²E/dV² = 2*b*V0

        V_eq = volumes[np.argmin(energies)]
        dV = volumes - V_eq

        # Fit polynomial
        coeffs = np.polyfit(dV, energies, 2)
        E0, a, b = coeffs[2], coeffs[1], coeffs[0]

        # Bulk modulus in GPa (convert from eV/Å³ to GPa)
        bulk_modulus_eV_per_A3 = 2 * b * V_eq
        bulk_modulus_GPa = (
            bulk_modulus_eV_per_A3 * 160.2176
        )  # Conversion factor

        # Create EOS plot
        V_fine = np.linspace(volumes.min(), volumes.max(), 100)
        dV_fine = V_fine - V_eq
        E_fine = E0 + a * dV_fine + b * dV_fine**2

        plt.figure(figsize=(10, 6))
        plt.scatter(
            volumes, energies, color="red", s=50, zorder=5, label="TB3 data"
        )
        plt.plot(V_fine, E_fine, "b-", linewidth=2, label="Quadratic fit")
        plt.axvline(
            V_eq,
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"V₀ = {V_eq:.3f} Å³",
        )
        plt.xlabel("Volume (Å³)")
        plt.ylabel("Energy (eV)")
        plt.title("Equation of State")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add text box with results
        textstr = f"B₀ = {bulk_modulus_GPa:.1f} GPa\nV₀ = {V_eq:.3f} Å³\nE₀ = {E0:.6f} eV"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        plt.text(
            0.02,
            0.98,
            textstr,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        plt.tight_layout()
        plt.savefig(work_path / "eos_fit.png", dpi=300, bbox_inches="tight")
        plt.close()

        t2 = time.time()
        # Compile results
        results = {
            "initial_volume_A3": float(V0),
            "equilibrium_volume_A3": float(V_eq),
            "equilibrium_energy_eV": float(E0),
            "bulk_modulus_GPa": float(bulk_modulus_GPa),
            "bulk_modulus_eV_per_A3": float(bulk_modulus_eV_per_A3),
            "volume_range": volume_range,
            "n_points_calculated": len(energies),
            "n_points_requested": n_points,
            "time": t2 - t1,
            "raw_data": {
                "volumes_A3": volumes.tolist(),
                "energies_eV": energies.tolist(),
            },
            "fit_parameters": {
                "E0": float(E0),
                "a": float(a),
                "b": float(b),
            },
            "plot_file": str(work_path / "eos_fit.png"),
        }

        # Save results
        with open(work_path / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=self._json_serializable)

        self.results["eos"] = results

        print(f"\n=== EOS Results Summary ===")
        print(f"Bulk Modulus: {bulk_modulus_GPa:.1f} GPa")
        print(f"Equilibrium Volume: {V_eq:.3f} Å³")
        print(f"Equilibrium Energy: {E0:.6f} eV")
        print(f"Results saved in {work_path}/")

        return results

    def run_phonon(
        self,
        atoms,
        work_dir="phonon",
        supercell=[2, 2, 2],
        amplitude=5e-4,
        tolerance=1e-4,
    ):
        """
        Run phonon calculation using Phonopy with TB3.

        Args:
            atoms: ASE Atoms object (optimized structure)
            work_dir: Working directory for phonon calculations
            supercell: Supercell dimensions
            amplitude: Displacement amplitude
            tolerance: Tolerance for symmetry detection

        Returns:
            Dictionary with phonon results
        """
        t1 = time.time()
        work_path = Path(work_dir)
        work_path.mkdir(exist_ok=True)

        # Write optimized structure
        poscar_content = self._atoms_to_poscar(atoms)
        self.write_poscar(poscar_content, "POSCAR", work_path)

        print("Setting up phonon calculation with Phonopy...")

        # Generate displaced structures
        dim_str = " ".join(map(str, supercell))
        phonopy_cmd = [
            "phonopy",
            "-d",
            f"--dim={dim_str}",
            f"--amplitude={amplitude}",
            f"--tolerance={tolerance}",
        ]

        try:
            self.run_command(phonopy_cmd, cwd=work_path)
        except Exception as e:
            print(f"Phonopy displacement generation failed: {e}")
            return {"error": "Phonopy displacement generation failed"}

        # Find displacement files
        disp_files = list(work_path.glob("POSCAR-*"))
        print(f"Found {len(disp_files)} displacement structures")

        if not disp_files:
            raise RuntimeError("No displacement files generated by Phonopy")

        # Calculate forces for each displacement
        force_files = []
        for disp_file in disp_files:
            disp_num = disp_file.name.split("-")[1]
            disp_dir = work_path / f"disp-{disp_num}"
            disp_dir.mkdir(exist_ok=True)

            print(f"Processing displacement {disp_num}...")

            # Copy displacement geometry
            shutil.copy(disp_file, disp_dir / "POSCAR")

            # Create TB3 force calculation script
            force_script = """
using ThreeBodyTB
using NPZ

println("Calculating forces for displaced structure...")
crys = makecrys("POSCAR")
#energy, tbc, flag = scf_energy(crys)
energy, forces, stress, tbc = scf_energy_force_stress(crys)
# Calculate forces 
#forces = ThreeBodyTB.TB.get_forces(tbc, crys)

println("Forces calculated.")
println("Max force: ", maximum(abs.(forces)))

# Save forces in FORCE_SETS format for Phonopy
open("FORCES", "w") do file
    natoms = size(forces, 1)
    for i in 1:natoms
        println(file, forces[i,1], " ", forces[i,2], " ", forces[i,3])
    end
end

# Also save as NPZ
npzwrite("forces.npz", forces)

println("Force calculation completed.")
"""

            try:
                self._run_julia_script(force_script, "forces.jl", disp_dir)
                force_files.append(str(disp_dir / "FORCES"))

            except Exception as e:
                print(f"  Error calculating forces: {e}")

        # Process forces with Phonopy
        if len(force_files) > 0:
            print("Processing forces with Phonopy...")
            phonopy_force_cmd = ["phonopy", "-f"] + force_files

            try:
                self.run_command(phonopy_force_cmd, cwd=work_path)

                # Calculate phonon properties
                phonon_results = self._calculate_phonon_properties(
                    work_path, atoms
                )

            except Exception as e:
                print(f"Phonopy force processing failed: {e}")
                phonon_results = {"error": "Phonopy force processing failed"}
        else:
            phonon_results = {"error": "No successful force calculations"}

        t2 = time.time()
        try:
            phonon_results["time"] = t2 - t1
        except Exception:
            pass
        # Save results
        with open(work_path / "results.json", "w") as f:
            json.dump(
                phonon_results, f, indent=2, default=self._json_serializable
            )

        self.results["phonon"] = phonon_results
        print(f"Phonon calculation completed. Results saved in {work_path}/")

        return phonon_results

    def _calculate_phonon_properties(self, work_path, atoms):
        """Calculate phonon band structure, DOS, and thermodynamic properties."""
        try:
            import phonopy
            from phonopy import Phonopy
            from phonopy.file_IO import parse_FORCE_SETS
            from phonopy.structure.atoms import PhonopyAtoms
        except ImportError:
            raise ImportError(
                "Phonopy is required for phonon calculations. Install with: pip install phonopy"
            )

        results = {}

        try:
            # Create Phonopy object
            unitcell = PhonopyAtoms(
                symbols=atoms.get_chemical_symbols(),
                scaled_positions=atoms.get_scaled_positions(),
                cell=atoms.get_cell(),
            )

            # Read supercell dimensions from phonopy_disp.yaml if it exists
            supercell_matrix = np.eye(3) * 2  # Default [2,2,2]
            if (work_path / "phonopy_disp.yaml").exists():
                try:
                    ph = phonopy.load(
                        work_path / "phonopy_disp.yaml",
                        force_sets_filename=work_path / "FORCE_SETS",
                    )
                except:
                    ph = Phonopy(unitcell, supercell_matrix)
                    if (work_path / "FORCE_SETS").exists():
                        force_sets = parse_FORCE_SETS(work_path / "FORCE_SETS")
                        ph.set_force_sets(force_sets)
            else:
                ph = Phonopy(unitcell, supercell_matrix)
                if (work_path / "FORCE_SETS").exists():
                    force_sets = parse_FORCE_SETS(work_path / "FORCE_SETS")
                    ph.set_force_sets(force_sets)

            # Produce force constants
            ph.produce_force_constants()

            # Calculate phonon band structure
            band_paths = self._get_phonon_band_paths(atoms)
            if band_paths:
                ph.run_band_structure(band_paths, is_eigenvectors=True)
                bs_dict = ph.get_band_structure_dict()

                # Create phonon band structure plot
                self._plot_phonon_bands(
                    bs_dict, work_path / "phonon_bands.png"
                )

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
            print(
                f"  Structure is {'stable' if min_freq > -1e-3 else 'unstable'}"
            )

        except Exception as e:
            print(f"Error in phonon calculations: {e}")
            results["error"] = str(e)

        return results

    def _get_phonon_band_paths(self, atoms):
        """Get high-symmetry k-point paths for phonon band structure."""
        try:
            # Simple cubic path as example - you may want to use spglib for automatic paths
            band_paths = [
                [
                    [0.0, 0.0, 0.0],  # Gamma
                    [0.5, 0.0, 0.0],  # X
                    [0.5, 0.5, 0.0],  # M
                    [0.0, 0.0, 0.0],  # Gamma
                    [0.5, 0.5, 0.5],  # R
                ]
            ]
            return band_paths
        except:
            return None

    def _plot_phonon_bands(self, bs_dict, output_file):
        """Create phonon band structure plot."""
        fig, ax = plt.subplots(figsize=(10, 6))

        distances = bs_dict["distances"]
        frequencies = bs_dict["frequencies"]

        # Plot each band
        for i in range(frequencies.shape[1]):
            ax.plot(
                distances, frequencies[:, i], "b-", linewidth=1.0, alpha=0.8
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

    def _atoms_to_poscar(self, atoms):
        """Convert ASE Atoms object to POSCAR string format."""
        from io import StringIO
        from ase.io.vasp import write_vasp

        poscar_io = StringIO()
        write_vasp(poscar_io, atoms, direct=True, vasp5=True)
        return poscar_io.getvalue()

    def _format_kpoints_for_tb3(self, kpoints):
        """Format k-points for TB3 Julia script."""
        # Convert k-points list to Julia array format
        kpath_lines = []
        for kpt in kpoints:
            kpath_lines.append(f"[{kpt[0]:.6f}, {kpt[1]:.6f}, {kpt[2]:.6f}]")
        kpath_str = f"kpath = [{', '.join(kpath_lines)}]"
        return kpath_str

    def _compare_with_vasp(self, work_path, vasprun, tb3_bands, tb3_fermi):
        """Compare TB3 band structure with VASP."""
        try:
            # Get VASP data
            vasp_fermi = vasprun.efermi
            vasp_eigs = (
                np.array([eig[:, 0] for eig in vasprun.eigenvalues[0]]).T
                - vasp_fermi
            )

            # Compare bands in energy window
            energy_tol = 4
            differences = []

            for k in range(min(tb3_bands.shape[0], vasp_eigs.shape[1])):
                for tb3_e in tb3_bands[k]:
                    if -energy_tol < tb3_e < energy_tol:
                        vasp_band = vasp_eigs[:, k]
                        valid_vasp = vasp_band[
                            (vasp_band > -energy_tol)
                            & (vasp_band < energy_tol)
                        ]
                        if len(valid_vasp) > 0:
                            min_diff = np.min(np.abs(tb3_e - valid_vasp))
                            differences.append(min_diff)

            max_diff = max(differences) if differences else 0

            # Create comparison plot
            fig = plt.figure(figsize=(12, 5))
            gs = GridSpec(1, 2)

            plt.subplot(gs[0, 0])
            plt.title("(a) Band Structure Comparison")
            for band in tb3_bands.T:
                plt.plot(
                    band,
                    "b-",
                    alpha=0.7,
                    label=(
                        "TB3"
                        if "TB3"
                        not in plt.gca().get_legend_handles_labels()[1]
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
                        if "VASP"
                        not in plt.gca().get_legend_handles_labels()[1]
                        else ""
                    ),
                )
            plt.ylim(-energy_tol, energy_tol)
            plt.ylabel("Energy [eV]")
            plt.legend()

            plt.subplot(gs[0, 1])
            plt.title("(b) Error Distribution")
            if differences:
                plt.scatter(
                    differences,
                    [tb3_bands.flatten()[i] for i in range(len(differences))],
                    alpha=0.6,
                )
            plt.ylim(-energy_tol, energy_tol)
            plt.xlabel("TB3 - VASP (δ) [eV]")
            plt.ylabel("Energy [eV]")

            plt.tight_layout()
            plt.savefig(
                work_path / "vasp_comparison.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

            return {
                "max_difference": max_diff,
                "mean_difference": np.mean(differences) if differences else 0,
                "comparison_plot": str(work_path / "vasp_comparison.png"),
                "n_compared_bands": len(differences),
            }

        except Exception as e:
            print(f"VASP comparison failed: {e}")
            return {"error": str(e)}

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

    def get_results(self) -> Dict:
        """Return all calculation results."""
        return self.results

    def cleanup(self, keep_essential: bool = True):
        """
        Clean up working directories.

        Args:
            keep_essential: Keep essential files like structures, results, plots
        """
        if keep_essential:
            keep_files = [
                "POSCAR",
                "POSCAR_relaxed",
                "tbc.xml.gz",
                "tbc_relaxed.xml.gz",
                "results.json",
                "electronic_properties.json",
                "*.png",
                "*.npz",
            ]
        else:
            keep_files = ["results.json"]

        for calc_dir in ["opt", "band", "eos", "phonon"]:
            calc_path = Path(calc_dir)
            if calc_path.exists():
                for item in calc_path.iterdir():
                    if item.is_file() and not any(
                        item.match(pattern) for pattern in keep_files
                    ):
                        try:
                            item.unlink()
                        except OSError:
                            pass


def main(jid="JVASP-816", julia_executable="julia", config=None):
    """
    Main execution function for TB3Py calculations.

    Args:
        jid: JARVIS ID for the material
        julia_executable: Path to Julia executable
        config: Configuration dictionary specifying which calculations to run
    """
    cwd = os.getcwd()
    work_path = Path(jid + "_tb3")
    work_path.mkdir(exist_ok=True)
    os.chdir(work_path)

    try:
        # Check Julia executable
        if not shutil.which(julia_executable):
            print(f"Error: Julia executable not found: {julia_executable}")
            print("Please install Julia or check the path")
            return

        # Download JARVIS data

        print(f"Downloading data for {jid}...")
        try:
            info = download_jarvis_dft_data(jid)
            atoms = info["atoms"]
            vasprun_bands = info["vasprun_bands"]
            kpoints_bands = info["kpoints_bands"]
            k_mesh = info["kpoints_scf"]
        except Exception as e:
            print(f"Error downloading VASP data: {e}")
            return

        # try:
        #    atoms, vasprun, kpoints = download_vasp_data(jid)
        #    print(f"Downloaded structure: {atoms.get_chemical_formula()}")
        # except Exception as e:
        #    print(f"Error downloading VASP data: {e}")
        #    return

        # Initialize calculator
        try:
            calc = TB3PyCalc(julia_executable=julia_executable, work_dir=".")
        except Exception as e:
            print(f"Error initializing calculator: {e}")
            return

        # Run optimization
        # if config.get("run_optimization", True):
        if "optimize_geometry" in config.properties_to_calculate:
            print("\n" + "=" * 50)
            print("Running geometry optimization...")
            print("=" * 50)
            try:
                opt_results, final_atoms = calc.run_optimization(atoms)
                print(f"✓ Final energy: {opt_results['energy']:.4f} eV")
                print(f"✓ Fermi level: {opt_results['fermi_level']:.4f} eV")
                print(
                    f"✓ Max force: {opt_results.get('max_force', 'N/A'):.4f} eV/Å"
                )
                print(f"✓ Results saved: opt/results.json")
            except Exception as e:
                print(f"✗ Error during optimization: {e}")
                final_atoms = atoms  # Use original structure

        else:
            final_atoms = atoms
        if "calculate_band_structure" in config.properties_to_calculate:
            # if config.get("run_band_structure", True):
            print("\n" + "=" * 50)
            print("Running band structure calculation...")
            print("=" * 50)
            try:
                vasprun_for_comparison = True
                #(
                #    vasprun_bands
                #    if config.get("compare_with_vasp", True)
                #    else None
                #)
                bs_path = ""
                for kk in kpoints_bands:
                    # bs_path+=map(str(" ".join(kk)))+";"
                    bs_path += " ".join(map(str, kk)) + ";"
                # print("bs_path",bs_path)
                band_results = calc.run_band_structure(
                    final_atoms,
                    "[" + bs_path + "]",
                    vasprun_for_comparison,
                    work_dir="band",
                    # final_atoms, kpoints, vasprun_for_comparison, work_dir="band"
                )
                print(
                    f"✓ Bandgap: {band_results.get('bandgap', 'N/A'):.4f} eV"
                )
                if band_results.get("vbm") is not None:
                    print(f"✓ VBM: {band_results['vbm']:.4f} eV")
                if band_results.get("cbm") is not None:
                    print(f"✓ CBM: {band_results['cbm']:.4f} eV")
                if band_results.get("vasp_comparison"):
                    comp = band_results["vasp_comparison"]
                    if "max_difference" in comp:
                        print(
                            f"✓ Max band difference vs VASP: {comp['max_difference']:.4f} eV"
                        )
                print(f"✓ Results saved: band/results.json")
            except Exception as e:
                print(f"✗ Error during band structure calculation: {e}")

        # Run EOS calculation
        if "calculate_eos" in config.properties_to_calculate:
            # if config.get("run_eos", True):
            print("\n" + "=" * 50)
            print("Running equation of state calculation...")
            print("=" * 50)
            try:
                eos_results = calc.run_eos_bulkmod(
                    final_atoms, work_dir="eos", volume_range=0.15, n_points=9
                )
                print(
                    f"✓ Bulk modulus: {eos_results['bulk_modulus_GPa']:.1f} GPa"
                )
                print(
                    f"✓ Equilibrium volume: {eos_results['equilibrium_volume_A3']:.3f} Å³"
                )
                print(f"✓ Results saved: eos/results.json")
            except Exception as e:
                print(f"✗ Error during EOS calculation: {e}")

        # Run phonon calculation
        if "calculate_phonons" in config.properties_to_calculate:
            print("\n" + "=" * 50)
            print("Running phonon calculation...")
            print("=" * 50)
            try:
                phonon_results = calc.run_phonon(
                    final_atoms,
                    work_dir="phonon",
                    supercell=[2, 2, 2],
                    amplitude=5e-4,
                    tolerance=1e-4,
                )
                if "error" not in phonon_results:
                    print(f"✓ Phonon calculation completed")
                    if "analysis" in phonon_results:
                        analysis = phonon_results["analysis"]
                        print(
                            f"✓ Min frequency: {analysis['min_frequency_THz']:.3f} THz"
                        )
                        print(
                            f"✓ Structure is {'stable' if analysis['is_stable'] else 'unstable'}"
                        )
                    print(f"✓ Results saved: phonon/results.json")
                else:
                    print(f"✗ Phonon error: {phonon_results['error']}")
            except Exception as e:
                print(f"✗ Error during phonon calculation: {e}")

        # Run band structure
        # Print summary
        print("\n" + "=" * 50)
        print("CALCULATION SUMMARY")
        print("=" * 50)
        print(f"System: {jid} ({atoms.get_chemical_formula()})")

        all_results = calc.get_results()

        if "optimization" in all_results:
            opt = all_results["optimization"]
            print(f"Final energy: {opt.get('energy', 'N/A'):.4f} eV")
            print(f"Fermi level: {opt.get('fermi_level', 'N/A'):.4f} eV")

        if "eos" in all_results:
            eos = all_results["eos"]
            print(
                f"Bulk modulus: {eos.get('bulk_modulus_GPa', 'N/A'):.1f} GPa"
            )

        if "band_structure" in all_results:
            bands = all_results["band_structure"]
            print(f"Bandgap: {bands.get('bandgap', 'N/A'):.4f} eV")

        print(f"\n✓ All calculations completed for {jid}!")

    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    # Example usage with different configurations

    # Full calculation suite
    main(jid="JVASP-816", julia_executable="julia")
