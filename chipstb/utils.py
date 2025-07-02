import os
import json
import shutil
import subprocess
import tempfile
import zipfile
import io
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import requests
import re
from jarvis.core.kpoints import Kpoints3D
from ase.io import read, write
from jarvis.core.atoms import Atoms, ase_to_atoms
from jarvis.db.figshare import get_jid_data
from jarvis.io.vasp.outputs import Vasprun



def download_jarvis_dft_data(jid):
    """Download DFT data from JARVIS database."""
    kpoints_bands=None
    kpoints_scf=None
    atoms=None
    vrun_bands=None
    dat = get_jid_data(jid=jid, dataset="dft_3d")
    atoms = Atoms.from_dict(dat["atoms"])
    lattice_mat=atoms.lattice_mat
    # atoms = atoms.get_conventional_atoms
    atoms = atoms.ase_converter()
    # TODO: Take full kpoints
    length=int(dat["kpoint_length_unit"]/4)

    kp = Kpoints3D().automatic_length_mesh(
        lattice_mat=lattice_mat, length=length
    )
    kpoints_scf = kp._kpoints[0]
    print("kpoints_scf ",kpoints_scf)
    info={}

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

            vrun_bands = Vasprun(path)
            kpoints_bands = extract_kpoints_from_vasprun(path)
        #if "DFT-SCF" in raw_file:
        #    calc_zipfile_link = raw_file.split(",")[2]
        #    r = requests.get(calc_zipfile_link)
        #    z = zipfile.ZipFile(io.BytesIO(r.content))
        #    vrun_content = z.read("vasprun.xml").decode("utf-8")

        #    # Create temporary file
        #    fd, path = tempfile.mkstemp()
        #    with os.fdopen(fd, "w") as tmp:
        #        tmp.write(vrun_content)

        #    vrun = Vasprun(path)


    info["atoms"]=atoms
    info["vasprun_bands"]=vrun_bands
    info["kpoints_bands"]=kpoints_bands
    info["kpoints_scf"]=kpoints_scf

    return info


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
