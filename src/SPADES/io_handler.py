import enum
import struct
import token
import numpy as np
from . import ph


def write_scattring_wf(file_name, kappa: list[int], ke_grid: np.ndarray, inner_grid: dict, coulomb_grid: dict, r_values: np.ndarray, p: dict, q: dict):
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


def read_scattering_wf(file_name):
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


def write_bound_wf(file_name, r_grid: np.ndarray, be_values: dict, p_values: dict, q_values: dict):
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
    with open(file_name, 'rb') as f:
        n_shells, nr = struct.unpack('<ii', f.read(8))
        be_values = {}
        p_values = {}
        q_values = {}
        r_grid = np.zeros(nr)

        for _ in n_shells:
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
                    p_values[n][k] = p
                    q_values[n][k] = q

    return r_grid, be_values, p_values, q_values


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
    n_fermi_functions = len(spectra)
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
                    print(type(psfs[ff_type][sp_type]))
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
    with open(filename, 'r') as f:
        lines = f.readlines()

    # determine number of header lines
    for i, line in enumerate(lines):
        if "Spectra:" in line:
            break
    n_header_lines = i+1

    # read header
    data = {}
    psfs_order = {}
    for i, line in enumerate(lines[:n_header_lines]):
        tokens = line.split("=")
        if "Parent nucleus" in line:
            data["parent_nucleus"] = tokens[1]
        if "Porcess = " in line:
            data["process"] = tokens[1]
        if "Energy unit = " in line:
            data["energy_unit"] = tokens[1]
        if "PSF unit = " in line:
            data["psf_unit"] = tokens[1]
        if "emin =" in line:
            data["emin"] = float(tokens[1])
        if "PSFs:" in line:
            data["PSFs"] = {}
            tokens_psfs = lines[i+1].strip().split()
            i_order = 0
            for tok in tokens_psfs:
                toks = tok.split('(')
                key_ff = toks[1].strip(')')
                sp_type = toks[0].strip()
                psfs_order[i_order] = (key_ff, sp_type)
                if key_ff in data["PSFs"]:
                    data["PSFs"][key_ff][sp_type] = 0.
                else:
                    data["PSFs"][key_ff] = {sp_type: 0.}
                i_order += 1

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

    for i in range(len(psfs_order)):
        key_ff, sp_type = psfs_order[i]
        if key_ff in data["Spectra"]:
            data["Spectra"][key_ff][sp_type] = np.zeros(data["n_points"])
        else:
            data["Spectra"][key_ff] = {sp_type: np.zeros(data["n_points"])}

    for i, line in enumerate(lines[n_header_lines+1:]):
        tokens_values = line.strip().split()
        data["energy_grid"][ie] = float(tokens_values[0])
        for j, tok in enumerate(tokens_values[1:]):
            key_ff, sp_type = psfs_order[j]
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
    lines = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    # determine number of header lines
    for i, line in enumerate(lines):
        if "Spectra:" in line:
            break
    n_header_lines = i+1

    # read header
    data = {}
    psfs_order = {}
    for i, line in enumerate(lines[:n_header_lines]):
        tokens = line.split("=")
        if "Parent nucleus" in line:
            data["parent_nucleus"] = tokens[1]
        if "Porcess = " in line:
            data["process"] = tokens[1]
        if "Energy unit = " in line:
            data["energy_unit"] = tokens[1]
        if "PSF unit = " in line:
            data["psf_unit"] = tokens[1]
        if "emin =" in line:
            data["emin"] = float(tokens[1])
        if "PSFs:" in line:
            data["PSFs"] = {}
            tokens_psfs = lines[i+1].strip().split()
            i_order = 0
            for tok in tokens_psfs:
                toks = tok.split('(')
                key_ff = toks[1].strip(')')
                sp_type = toks[0].strip()
                psfs_order[i_order] = (key_ff, sp_type)
                if key_ff in data["PSFs"]:
                    data["PSFs"][key_ff][sp_type] = 0.
                else:
                    data["PSFs"][key_ff] = {sp_type: 0.}
                i_order += 1

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
    for i in range(len(psfs_order)):
        key_ff, sp_type = psfs_order[i]
        if key_ff in data["Spectra"]:
            data["Spectra"][key_ff][sp_type] = np.zeros_like(data["e1_grid"])
        else:
            data["Spectra"][key_ff] = {sp_type:
                                       np.zeros_like(data["e1_grid"])}

    for i, line in enumerate(lines[n_header_lines+1:]):
        tokens_values = line.strip().split()
        data["e1_grid"][ie, je] = float(tokens_values[0])
        data["e2_grid"][ie, je] = float(tokens_values[1])
        for j, tok in enumerate(tokens_values[2:]):
            key_ff, sp_type = psfs_order[j]
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
    n_fermi_functions = len(ff0)

    with open(file_name, "w") as f:
        f.write(header_fermi_functions.format(parent_nucleus=parent_nucleus,
                                              process=process,
                                              energy_unit=ph.user_energy_unit_name,
                                              n_energy_points=len(e_grid)))

        line = f'{"E":>10s}'
        for ff_type in ff0:
            line = line+f'{'FF0(' + ph.FERMIFUNCTIONS_NICE[ff_type]+')':>25s}'
        if not (ff1 is None):
            for ff_type in ff1:
                line = line + \
                    f'{'FF1(' + ph.FERMIFUNCTIONS_NICE[ff_type]+')':>25s}'

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
