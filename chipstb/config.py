from typing import List, Optional, Dict
from pydantic_settings import BaseSettings
from pydantic import Field


class CHIPSTBConfig(BaseSettings):
    structure_path: Optional[str] = Field(
        default=None,
        description="Local file path to a CIF, POSCAR, or other structure file",
    )
    jid_list: Optional[List[str]] = None

    calculator_types: Optional[List[str]] = Field(
        default=None,
        description="List of calculator types for ensemble or benchmarking",
    )

    dftb_executable: Optional[str] = Field(
        default=None,
        description="DFTB+ path, try 'which dftb+'",
    )
    # Path to Slater-Koster or Hamiltonian data
    skf_dir: Optional[str] = Field(
        default=None,
        description="Path to SK files",
    )

    # General TB settings
    # k_mesh: Optional[List[List[int]]] = Field(
    #    default_factory=lambda: [[6, 6, 6]],
    #    description="List of k-meshes for each material in jid_list or structure",
    # )

    band_kpoints_path: Optional[str] = Field(
        default=None,
        description="File path to high-symmetry k-points for band structure calculation",
    )

    # Optional settings for geometry optimization
    optimize_geometry: bool = Field(
        default=True, description="Whether to perform geometry optimization"
    )

    # Electronic structure settings
    calculate_dos: bool = Field(
        default=True, description="Whether to calculate the density of states"
    )
    calculate_band_structure: bool = Field(
        default=True, description="Whether to calculate the band structure"
    )

    # TB-specific control (overridable per calculator)
    calculator_settings: Dict[str, Dict] = Field(
        default_factory=dict,
        description="Dictionary of calculator-specific settings (e.g., parameters for DFTB+, xTB)",
    )

    # Optional comparison with external references
    compare_with_vasp: bool = Field(
        default=False,
        description="Compare TB band structure with VASP (requires vasprun.xml)",
    )
    vasprun_path: Optional[str] = Field(
        default=None,
        description="Path to VASP vasprun.xml for band structure comparison",
    )

    # Output control
    output_dir: str = Field(
        default="tb_output",
        description="Directory where output files will be saved",
    )
    plot_energy_range: List[float] = Field(
        default_factory=lambda: [-6.0, 6.0],
        description="Energy range in eV for band structure and DOS plots",
    )

    # What to compute
    properties_to_calculate: List[str] = Field(
        default_factory=lambda: [
            "optimize_geometry",
            "calculate_dos",
            "calculate_eos",
            "calculate_phonons",
            "calculate_band_structure",
            "compare_with_vasp",
        ],
        description="List of properties or analyses to run",
    )
