"""Serialization helpers for wavefunctions, spectra, PSFs, and Fermi functions."""

import struct
import numpy as np
from . import ph
from .data_models import BoundWavefunctions, ScatteringWavefunctions


def write_scattering_wf(file_name, kappa: list[int], ke_grid: np.ndarray, inner_grid: dict, coulomb_grid: dict, r_values: np.ndarray, p: dict, q: dict):
    """Write scattering wavefunctions to a compact binary format.

    Parameters
    ----------
    file_name:
        Output binary file path.
    kappa:
        List of relativistic angular quantum numbers.
    ke_grid:
        Kinetic-energy grid.
    inner_grid, coulomb_grid:
        Dictionaries of inner and Coulomb phase shifts keyed by ``kappa``.
    r_values:
        Radial grid values.
    p, q:
        Dictionaries containing large/small wavefunction components by ``kappa`` and energy index.
    """
    with open(file_name, 'wb') as f:
        f.write(struct.pack('<i', len(kappa)))
        f.write(struct.pack('<i', len(ke_grid)))
        f.write(struct.pack('<i', len(r_values)))

        for k in kappa:
            f.write(struct.pack('<i', k))
        for ke in ke_grid:
            f.write(struct.pack('<d', ke))
        for r in r_values:
            f.write(struct.pack('<d', r))

        for k in kappa:
            for ie in range(len(ke_grid)):
                f.write(struct.pack(
                    '<dd', inner_grid[k][ie], coulomb_grid[k][ie]))
                for ir in range(len(r_values)):
                    f.write(struct.pack('<dd', p[k][ie][ir], q[k][ie][ir]))


def write_scattring_wf(file_name, kappa: list[int], ke_grid: np.ndarray, inner_grid: dict, coulomb_grid: dict, r_values: np.ndarray, p: dict, q: dict):
    """Backward-compatible alias for :func:`write_scattering_wf`.

    Parameters
    ----------
    file_name, kappa, ke_grid, inner_grid, coulomb_grid, r_values, p, q:
        Forwarded unchanged to :func:`write_scattering_wf`.
    """
    write_scattering_wf(file_name, kappa, ke_grid, inner_grid, coulomb_grid, r_values, p, q)


def read_scattering_wf(file_name):
    """Read scattering wavefunctions from the binary format written by ``write_scattering_wf``.

    Parameters
    ----------
    file_name:
        Input binary file path.

    Returns
    -------
    tuple
        ``(k_values, e_values, r_values, inner_phase_values, coulomb_values, p_values, q_values)``.
    """
    with open(file_name, 'rb') as f:
        nk = struct.unpack('<i', f.read(4))[0]
        ne = struct.unpack('<i', f.read(4))[0]
        nr = struct.unpack('<i', f.read(4))[0]

        k_values = []
        for _ in range(nk):
            k_values.append(struct.unpack('<i', f.read(4))[0])

        e_values = np.zeros(ne)
        for ie in range(ne):
            e_values[ie] = struct.unpack('<d', f.read(8))[0]

        r_values = np.zeros(nr)
        for ir in range(nr):
            r_values[ir] = struct.unpack('<d', f.read(8))[0]

        inner_phase_values = {}
        coulomb_values = {}
        p_values = {}
        q_values = {}

        for k in k_values:
            inner_phase_values[k] = np.zeros(ne)
            coulomb_values[k] = np.zeros(ne)

            p_values[k] = np.zeros((ne, nr))
            q_values[k] = np.zeros_like(p_values[k])
            for ie in range(ne):
                delta_i, delta_c = struct.unpack('<dd', f.read(16))
                inner_phase_values[k][ie] = delta_i
                coulomb_values[k][ie] = delta_c

                for ir in range(nr):
                    p, q = struct.unpack('<dd', f.read(16))

                    p_values[k][ie][ir] = p
                    q_values[k][ie][ir] = q

    return (k_values, e_values, r_values, inner_phase_values, coulomb_values, p_values, q_values)


def read_scattering_wf_data(file_name: str) -> ScatteringWavefunctions:
    """Read scattering wavefunctions and return a typed data container.

    Parameters
    ----------
    file_name:
        Input binary file path.

    Returns
    -------
    ScatteringWavefunctions
        Structured scattering-wavefunction container.
    """
    k, e, r, inner, coul, p, q = read_scattering_wf(file_name)
    return ScatteringWavefunctions(
        k_values=k,
        energy_grid=e,
        radial_grid=r,
        inner_phase_values=inner,
        coulomb_phase_values=coul,
        p_values=p,
        q_values=q,
    )


def write_bound_wf(file_name, r_grid: np.ndarray, be_values: dict, p_values: dict, q_values: dict):
    """Write bound-state radial functions and binding energies to binary.

    Parameters
    ----------
    file_name:
        Output binary file path.
    r_grid:
        Radial grid.
    be_values:
        Nested dictionary of binding energies by principal quantum number and ``kappa``.
    p_values, q_values:
        Nested dictionaries of large/small components sampled on ``r_grid``.
    """
    nr = len(r_grid)
    n_shells = 0
    for n in be_values:
        for _ in be_values[n]:
            n_shells = n_shells+1

    with open(file_name, 'wb') as f:
        f.write(struct.pack('<ii', n_shells, nr))

        for n in be_values:
            for k in be_values[n]:
                f.write(struct.pack('<iid', n, k, be_values[n][k]))

        for ir in range(nr):
            f.write(struct.pack('<d', r_grid[ir]))

        for n in be_values:
            for k in be_values[n]:
                for ir in range(nr):
                    f.write(struct.pack(
                        '<dd', p_values[n][k][ir], q_values[n][k][ir]))


def read_bound_wf(file_name):
    """Read bound-state binary data written by :func:`write_bound_wf`.

    Parameters
    ----------
    file_name:
        Input binary file path.

    Returns
    -------
    tuple
        ``(r_grid, be_values, p_values, q_values)``.
    """
    with open(file_name, 'rb') as f:
        n_shells, nr = struct.unpack('<ii', f.read(8))
        be_values = {}
        p_values = {}
        q_values = {}
        r_grid = np.zeros(nr)

        for _ in range(n_shells):
            n, k, be = struct.unpack('<iid', f.read(16))
            try:
                be_values[n][k] = be
            except KeyError:
                be_values[n] = {}
                be_values[n][k] = be

        for ir in range(nr):
            r_grid[ir] = struct.unpack('<d', f.read(8))[0]

        for n in be_values:
            p_values[n] = {}
            q_values[n] = {}
            for k in be_values[n]:
                p_values[n][k] = np.zeros(nr)
                q_values[n][k] = np.zeros(nr)

                for ir in range(nr):
                    p, q = struct.unpack("<dd", f.read(16))
                    p_values[n][k][ir] = p
                    q_values[n][k][ir] = q

    return r_grid, be_values, p_values, q_values


def read_bound_wf_data(file_name: str) -> BoundWavefunctions:
    """Read bound wavefunctions and return a typed data container.

    Parameters
    ----------
    file_name:
        Input binary file path.

    Returns
    -------
    BoundWavefunctions
        Structured bound-wavefunction container.
    """
    r, be, p, q = read_bound_wf(file_name)
    return BoundWavefunctions(
        radial_grid=r,
        binding_energies=be,
        p_values=p,
        q_values=q,
    )


header_spectra = '''# Spectra and PSF values computed with project_name version
# Parent nucleus = {parent_nucleus}
# Process = {process}
# Energy unit = {energy_unit}
# PSF unit = {psf_unit}
# N energy points = {n_energy_points}
# Single (Summed) electron energy spectra are normalized to unit integral
# Legend:
# - dG/de = single electron energy spectrum
# - dH/de = single electron energy spectrum, angular part
# - dG/dT = summed electron energy spectrum
# - alpha = (dH/de)/(dG/de), angular correlation spectrum
# - G     = Total PSF
# - H     = Angular PSF
# - K     = Angular correlation coefficient
'''


def write_spectra(file_name, parent_nucleus: str, process: str, e_grid: np.ndarray, spectra: dict, psfs: dict | None = None):
    """Write 1D spectra and optional PSFs in a human-readable text format.

    Parameters
    ----------
    file_name:
        Output text file path.
    parent_nucleus:
        Parent nucleus label.
    process:
        Process identifier string.
    e_grid:
        1D kinetic-energy grid.
    spectra:
        Nested spectra dictionary keyed by Fermi-function and spectrum names.
    psfs:
        Optional nested PSF dictionary.
    """
    with open(file_name, "w") as f:
        f.write(header_spectra.format(parent_nucleus=parent_nucleus,
                                      process=process,
                                      energy_unit=ph.user_energy_unit_name,
                                      psf_unit=ph.user_psf_unit_name,
                                      n_energy_points=len(e_grid)))

        if not (psfs is None):
            f.write("# PSFs:\n")
            line = ""
            for ff_type in psfs:
                for sp_type in psfs[ff_type]:
                    if (type(psfs[ff_type][sp_type]) is dict):
                        for ord in psfs[ff_type][sp_type]:
                            line = line +\
                                f'{sp_type+"_"+ord+'('+ff_type+')':>25s}'
                    else:
                        line = line+f'{sp_type+'('+ff_type+')':>25s}'

            line = line+"\n"
            f.write(line)

            line = ""
            for ff_type in psfs:
                for sp_type in psfs[ff_type]:
                    if type(psfs[ff_type][sp_type]) is dict:
                        for ord in psfs[ff_type][sp_type]:
                            line = line+f'{psfs[ff_type][sp_type][ord]:25.13E}'
                    else:
                        line = line+f'{psfs[ff_type][sp_type]:25.13E}'

            line = line+"\n"
            f.write(line)

        f.write("# Spectra:\n")
        line = f"{'E':>25s}"
        for ff_type in spectra:
            for sp_type in spectra[ff_type]:
                if type(spectra[ff_type][sp_type]) is dict:
                    for ord in spectra[ff_type][sp_type]:
                        line = line+f'{sp_type+'_'+ord+'('+ff_type+')':>25s}'
                else:
                    line = line+f'{sp_type+'('+ff_type+')':>25s}'

        line = line+"\n"
        f.write(line)

        for ie in range(len(e_grid)):
            line = f"{e_grid[ie]:25.13E}"
            for ff_type in spectra:
                for sp_type in spectra[ff_type]:
                    if type(spectra[ff_type][sp_type]) is dict:
                        for ord in spectra[ff_type][sp_type]:
                            line = line +\
                                f'{spectra[ff_type][sp_type][ord][ie]:25.13E}'
                    else:
                        line = line+f'{spectra[ff_type][sp_type][ie]:25.13E}'
            line = line+"\n"
            f.write(line)


def load_data(filename):
    """Load spectra/PSF text files written by :func:`write_spectra`.

    Parameters
    ----------
    filename:
        Input text file path.

    Returns
    -------
    dict
        Parsed metadata, energy grid, spectra, and optional PSFs.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # determine location of spectra section
    i_spectra = -1
    for i, line in enumerate(lines):
        if "Spectra:" in line:
            i_spectra = i
            break
    if i_spectra < 0:
        raise ValueError("Could not find 'Spectra:' section in file")

    # read header
    data = {}
    psfs_order = []
    for i, line in enumerate(lines[:i_spectra+1]):
        tokens = line.split("=")
        if "Parent nucleus" in line:
            data["parent_nucleus"] = tokens[1].strip()
        if ("Process = " in line) or ("Porcess = " in line):
            data["process"] = tokens[1].strip()
        if "Energy unit = " in line:
            data["energy_unit"] = tokens[1].strip()
        if "PSF unit = " in line:
            data["psf_unit"] = tokens[1].strip()
        if "emin =" in line:
            data["emin"] = float(tokens[1])
        if "PSFs:" in line:
            data["PSFs"] = {}
            tokens_psfs = lines[i+1].strip().split()
            for tok in tokens_psfs:
                toks = tok.split('(')
                key_ff = toks[1].strip(')')
                sp_type = toks[0].strip()
                psfs_order.append((key_ff, sp_type))
                if key_ff in data["PSFs"]:
                    data["PSFs"][key_ff][sp_type] = 0.
                else:
                    data["PSFs"][key_ff] = {sp_type: 0.}

            tokens_values = lines[i+2].strip().split()
            for i, tok in enumerate(tokens_values):
                key_ff, sp_type = psfs_order[i]
                data["PSFs"][key_ff][sp_type] = float(tok)
        if "N energy points =" in line:
            toks = tokens[1].strip().split()
            data["n_points"] = int(toks[0])

    # read data
    ie = 0
    data["energy_grid"] = np.zeros(data["n_points"])
    data["Spectra"] = {}
    spectra_order = []
    spectra_labels = lines[i_spectra+1].strip().split()[1:]
    for label in spectra_labels:
        tok, ff_tok = label.split("(")
        ff_type = ff_tok.strip(")")
        sp_type = tok.strip()
        spectra_order.append((ff_type, sp_type))
        if ff_type in data["Spectra"]:
            data["Spectra"][ff_type][sp_type] = np.zeros(data["n_points"])
        else:
            data["Spectra"][ff_type] = {sp_type: np.zeros(data["n_points"])}

    for i, line in enumerate(lines[i_spectra+2:]):
        tokens_values = line.strip().split()
        if not tokens_values:
            continue
        data["energy_grid"][ie] = float(tokens_values[0])
        for j, tok in enumerate(tokens_values[1:]):
            key_ff, sp_type = spectra_order[j]
            data["Spectra"][key_ff][sp_type][ie] = float(tok)
        ie += 1
    return data


header_spectra_2D = '''# PSFs and 2D (e_1, e_2) spectra computed with project_name version
# Parent nucleus = {parent_nucleus}
# Process = {process}
# Energy unit = {energy_unit}
# PSF unit = {psf_unit}
# Rectangular grid: eta1 in [0, q_value-2*emin], eta2 in [0, 1]
# Transformation: e1 = eta1 + emin,
#                 e2 = eta2*(q_value - eta1 - 2*emin) + emin
# N points = {n_points_e1} {n_points_e2}
# Spectra are normalized to unit integral
# emin = {emin}
# Legend:
# - dG/deta1deta2 = angle-independent electron energy spectrum
# - dH/deta1deta2 = angle-dependent electron energy spectrum
'''


def write_2d_spectra(file_name, parent_nucleus: str, process: str, e1_grid: np.ndarray, e2_grid: np.ndarray, emin: float, spectra: dict, psfs: dict):
    """Write 2D spectra and PSFs on transformed ``(e1, e2)`` grids.

    Parameters
    ----------
    file_name:
        Output text file path.
    parent_nucleus:
        Parent nucleus label.
    process:
        Process identifier string.
    e1_grid, e2_grid:
        2D meshgrids used for spectra sampling.
    emin:
        Lower kinetic-energy threshold used by the transformation.
    spectra, psfs:
        Nested dictionaries with spectra arrays and PSF values.
    """
    with open(file_name, "w") as f:
        f.write(header_spectra_2D.format(
            parent_nucleus=parent_nucleus,
            process=process,
            energy_unit=ph.user_energy_unit_name,
            psf_unit=ph.user_psf_unit_name,
            n_points_e1=len(e1_grid), n_points_e2=len(e2_grid),
            emin=emin))

        f.write("# PSFs:\n")
        line = ""
        for ff_type in psfs:
            for sp_type in psfs[ff_type]:
                if type(psfs[ff_type][sp_type]) is dict:
                    for ord in psfs[ff_type][sp_type]:
                        line = line +\
                            f'{sp_type+"_"+ord+'('+ff_type+')':>25s}'
                else:
                    line = line+f'{sp_type+'('+ff_type+')':>25s}'

        line = line+"\n"
        f.write(line)

        line = ""
        for ff_type in psfs:
            for sp_type in psfs[ff_type]:
                if (type(psfs[ff_type][sp_type]) is dict):
                    for ord in psfs[ff_type][sp_type]:
                        line = line+f'{psfs[ff_type][sp_type][ord]:25.13E}'
                else:
                    line = line+f'{psfs[ff_type][sp_type]:25.13E}'
        line = line+"\n"
        f.write(line)

        f.write("# Spectra:\n")
        line = f"{'E1':>25s} {'E2':>25s}"
        for ff_type in spectra:
            for sp_type in spectra[ff_type]:
                if type(spectra[ff_type][sp_type]) is dict:
                    for ord in spectra[ff_type][sp_type]:
                        line = line+f'{sp_type+"_"+ord+'('+ff_type+')':>25s}'
                else:
                    line = line+f'{sp_type+'('+ff_type+')':>25s}'

        line = line+"\n"
        f.write(line)

        for ie in range(len(e1_grid)):
            for je in range(len(e2_grid)):
                e1 = e1_grid[ie][je]
                e2 = e2_grid[ie][je]
                line = f"{e1:25.13E} {e2:25.13E}"
                for ff_type in spectra:
                    for sp_type in spectra[ff_type]:
                        if type(spectra[ff_type][sp_type]) is dict:
                            for ord in spectra[ff_type][sp_type]:
                                spec = spectra[ff_type][sp_type][ord][ie, je]
                                line = line + f'{spec:25.13E}'
                        else:
                            spec = spectra[ff_type][sp_type][ie, je]
                            line = line+f'{spec:25.13E}'
                line = line+"\n"
                f.write(line)


def load_2d_spectra(filename):
    """Load 2D spectra files written by :func:`write_2d_spectra`.

    Parameters
    ----------
    filename:
        Input text file path.

    Returns
    -------
    dict
        Parsed metadata, 2D grids, spectra arrays, and PSFs.
    """
    lines = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    # determine location of spectra section
    i_spectra = -1
    for i, line in enumerate(lines):
        if "Spectra:" in line:
            i_spectra = i
            break
    if i_spectra < 0:
        raise ValueError("Could not find 'Spectra:' section in file")

    # read header
    data = {}
    psfs_order = []
    for i, line in enumerate(lines[:i_spectra+1]):
        tokens = line.split("=")
        if "Parent nucleus" in line:
            data["parent_nucleus"] = tokens[1].strip()
        if ("Process = " in line) or ("Porcess = " in line):
            data["process"] = tokens[1].strip()
        if "Energy unit = " in line:
            data["energy_unit"] = tokens[1].strip()
        if "PSF unit = " in line:
            data["psf_unit"] = tokens[1].strip()
        if "emin =" in line:
            data["emin"] = float(tokens[1])
        if "PSFs:" in line:
            data["PSFs"] = {}
            tokens_psfs = lines[i+1].strip().split()
            for tok in tokens_psfs:
                toks = tok.split('(')
                key_ff = toks[1].strip(')')
                sp_type = toks[0].strip()
                psfs_order.append((key_ff, sp_type))
                if key_ff in data["PSFs"]:
                    data["PSFs"][key_ff][sp_type] = 0.
                else:
                    data["PSFs"][key_ff] = {sp_type: 0.}

            tokens_values = lines[i+2].strip().split()
            for i, tok in enumerate(tokens_values):
                key_ff, sp_type = psfs_order[i]
                data["PSFs"][key_ff][sp_type] = float(tok)
        if "N points =" in line:
            toks = tokens[1].strip().split()
            data["n_points_e1"] = int(toks[0])
            data["n_points_e2"] = int(toks[1])

    # read data
    ie = 0
    je = 0
    data["e1_grid"] = np.zeros((data["n_points_e1"], data["n_points_e2"]))
    data["e2_grid"] = np.zeros((data["n_points_e1"], data["n_points_e2"]))
    data["Spectra"] = {}
    spectra_order = []
    spectra_labels = lines[i_spectra+1].strip().split()[2:]
    for label in spectra_labels:
        tok, ff_tok = label.split("(")
        key_ff = ff_tok.strip(")")
        sp_type = tok.strip()
        spectra_order.append((key_ff, sp_type))
        if key_ff in data["Spectra"]:
            data["Spectra"][key_ff][sp_type] = np.zeros_like(data["e1_grid"])
        else:
            data["Spectra"][key_ff] = {sp_type: np.zeros_like(data["e1_grid"])}

    for i, line in enumerate(lines[i_spectra+2:]):
        tokens_values = line.strip().split()
        if not tokens_values:
            continue
        data["e1_grid"][ie, je] = float(tokens_values[0])
        data["e2_grid"][ie, je] = float(tokens_values[1])
        for j, tok in enumerate(tokens_values[2:]):
            key_ff, sp_type = spectra_order[j]
            data["Spectra"][key_ff][sp_type][ie, je] = float(tok)
        je += 1
        if je == data["n_points_e2"]:
            je = 0
            ie += 1

    return data


header_fermi_functions = '''# Fermi functions computed with project_name version
# Parent nucleus = {parent_nucleus}
# Process = {process}
# Energy unit = {energy_unit}
# N energy points = {n_energy_points}
'''


def write_fermi_functions(file_name, parent_nucleus: str, process: str, e_grid: np.ndarray, ff0: dict, ff1: dict | None = None):
    """Write Fermi-function tables versus kinetic energy.

    Parameters
    ----------
    file_name:
        Output text file path.
    parent_nucleus:
        Parent nucleus label.
    process:
        Process identifier string.
    e_grid:
        1D kinetic-energy grid.
    ff0, ff1:
        Fermi-function values keyed by backend type.
    """
    n_fermi_functions = len(ff0)

    with open(file_name, "w") as f:
        f.write(header_fermi_functions.format(parent_nucleus=parent_nucleus,
                                              process=process,
                                              energy_unit=ph.user_energy_unit_name,
                                              n_energy_points=len(e_grid)))

        line = f'{"E":>10s}'
        for ff_type in ff0:
            line = line + \
                f'{'FF0(' + ph.FERMIFUNCTIONS_MAP_REV[ff_type]+')':>25s}'
        if not (ff1 is None):
            for ff_type in ff1:
                line = line + \
                    f'{'FF1(' + ph.FERMIFUNCTIONS_MAP_REV[ff_type]+')':>25s}'

        line = line + '\n'
        f.write(line)

        for ie in range(len(e_grid)):
            line = f"{e_grid[ie]:10.5f}"
            for ff_type in ff0:
                line = line+f'{ff0[ff_type][ie]:25.13E}'
            if not (ff1 is None):
                for ff_type in ff1:
                    line = line+f'{ff1[ff_type][ie]:25.13E}'
            line = line + '\n'
            f.write(line)


def load_fermi_functions(filename):
    """Load Fermi-function tables written by :func:`write_fermi_functions`.

    Parameters
    ----------
    filename:
        Input text file path.

    Returns
    -------
    dict
        Parsed energy grid and FF datasets.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    data = {}
    i_point = 0
    ff_types = {}
    for line in lines:
        if line[0] == "#":
            if "N energy points" in line:
                tokens = line.strip().split("=")
                data["energy_grid"] = np.zeros(int(tokens[1]))
                continue
            continue
        tokens = line.split()
        if tokens[0] == 'E':
            for i, tok in enumerate(tokens):
                if i == 0:
                    continue
                key_ff = tok
                ff_types[i] = key_ff
                data[key_ff] = np.zeros(len(data["energy_grid"]))
            continue
        else:
            for i, tok in enumerate(tokens):
                if i == 0:
                    data["energy_grid"][i_point] = float(tok)
                    continue
                toks = tok.split()
                key_ff = ff_types[i]
                data[key_ff][i_point] = float(toks[0])

            i_point += 1

    return data
