# Typical Slater-Koster (SKF) File Format

The Slater-Koster file (.skf) format is used in Density Functional Tight Binding (DFTB) calculations to store parametrized Hamiltonian and overlap matrix elements, atomic orbital information, and repulsive potentials.

## Overall Structure

A typical SKF file contains the following sections in order:

1. **Header Line**: Grid parameters
2. **Atomic Data Section**: Orbital energies, occupations, and Hubbard parameters
3. **Integral Tables**: Hamiltonian and overlap matrix elements
4. **Repulsive Potential**: Distance-dependent repulsive energy terms

## Detailed Format Description

### 1. Header Line
```
grid_distance  n_grid_points  [additional_info]
```
- `grid_distance`: Distance increment between grid points (typically in Bohr)
- `n_grid_points`: Number of grid points in the tables
- Additional fields may contain atomic numbers or other metadata

**Example:**
```
0.02  500
```

### 2. Atomic Data Section
This section describes the atomic orbitals and their properties. The format varies between parameter sets but typically includes:

```
Ed  Es  Ep  [Ed]  SpinW  Ud  Us  Up  [Ud]  fd  fs  fp  [fd]
```

Where:
- `Ed`, `Es`, `Ep`: Orbital energies for d, s, p orbitals (in Hartree)
- `SpinW`: Spin polarization constant (often 0.0 for non-spin-polarized)
- `Ud`, `Us`, `Up`: Hubbard parameters for d, s, p orbitals
- `fd`, `fs`, `fp`: Orbital occupations for d, s, p orbitals

**Example for Carbon (C-C.skf):**
```
-0.508853  -0.508853   0.0   -0.311054  -0.311054   0.0   2.0  2.0  0.0
```

### 3. Integral Tables

The integral tables contain the distance-dependent Hamiltonian and overlap matrix elements. These are organized as:

#### Table Structure:
- **Hamiltonian Table**: H matrix elements
- **Overlap Table**: S matrix elements

Each table contains rows corresponding to different orbital pair interactions:

#### Orbital Pair Types:
For different angular momentum combinations (s, p, d), the interactions are:

**s-s interactions:**
- σ (sigma) bond

**s-p interactions:**
- σ bond

**p-p interactions:**
- σ bond
- π (pi) bond

**s-d interactions:**
- σ bond

**p-d interactions:**
- σ bond
- π bond

**d-d interactions:**
- σ bond
- π bond
- δ (delta) bond

#### Table Format:
```
H_ss_sigma(r1)    H_ss_sigma(r2)    H_ss_sigma(r3)    ...
H_sp_sigma(r1)    H_sp_sigma(r2)    H_sp_sigma(r3)    ...
H_pp_sigma(r1)    H_pp_sigma(r2)    H_pp_sigma(r3)    ...
H_pp_pi(r1)       H_pp_pi(r2)       H_pp_pi(r3)       ...
...
S_ss_sigma(r1)    S_ss_sigma(r2)    S_ss_sigma(r3)    ...
S_sp_sigma(r1)    S_sp_sigma(r2)    S_sp_sigma(r3)    ...
...
```

### 4. Repulsive Potential
The repulsive potential section contains the short-range repulsive energy as a function of distance:

```
V_rep(r1)  V_rep(r2)  V_rep(r3)  V_rep(r4)  ...
```

This is typically a spline-fitted potential that decays to zero at large distances.

## Complete Example: Simplified C-C.skf File

```
0.02  500
-0.50885  -0.50885   0.0   -0.31105  -0.31105   0.0   2.0  2.0  0.0
# Hamiltonian matrix elements (ss_sigma, sp_sigma, pp_sigma, pp_pi)
-0.5000  -0.4950  -0.4900  -0.4850  -0.4800  ...  # H_ss_sigma
-0.4200  -0.4150  -0.4100  -0.4050  -0.4000  ...  # H_sp_sigma  
-0.3800  -0.3750  -0.3700  -0.3650  -0.3600  ...  # H_pp_sigma
 0.2500   0.2450   0.2400   0.2350   0.2300  ...  # H_pp_pi
# Overlap matrix elements (ss_sigma, sp_sigma, pp_sigma, pp_pi)
 1.0000   0.9950   0.9900   0.9850   0.9800  ...  # S_ss_sigma
 0.8500   0.8450   0.8400   0.8350   0.8300  ...  # S_sp_sigma
 0.7800   0.7750   0.7700   0.7650   0.7600  ...  # S_pp_sigma
 0.6200   0.6150   0.6100   0.6050   0.6000  ...  # S_pp_pi
# Repulsive potential
 5.2000   4.8500   4.5200   4.2100   3.9200  ...
```

## Important Notes

1. **Units**: 
   - Energies are typically in Hartree atomic units
   - Distances are in Bohr atomic units
   - Some parameter sets may use different units

2. **Parameter Set Variations**:
   - Different parameter sets (mio, pbc, 3ob, etc.) may have slightly different formats
   - The number of orbitals and interactions varies by element
   - Some files include spin-polarization terms

3. **File Naming Convention**:
   - Homonuclear: `X-X.skf` (e.g., `C-C.skf`)
   - Heteronuclear: `X-Y.skf` (e.g., `C-H.skf`)

4. **Grid Points**:
   - Typically 200-500 grid points
   - Grid spacing usually 0.02-0.05 Bohr
   - Maximum distance typically 10-15 Bohr

5. **Interpolation**:
   - Values between grid points are obtained by interpolation
   - Linear or spline interpolation is commonly used
   - Extrapolation beyond the grid should be avoided

This format allows DFTB codes to efficiently access pre-computed matrix elements during calculations, avoiding expensive integral evaluations at runtime.
