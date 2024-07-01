import struct
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


header_spectra = '''
# Spectra and PSF values computed with project_name version
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


def write_spectra(file_name, e_grid: np.ndarray, spectra: dict, psfs: dict | None = None):
    n_fermi_functions = len(spectra)
    with open(file_name, "w") as f:
        f.write(header_spectra.format(energy_unit=ph.user_energy_unit_name,
                                      psf_unit=ph.user_psf_unit_name,
                                      n_energy_points=len(e_grid)))

        if not (psfs is None):
            f.write("# PSFs:\n")
            line = ""
            for ff_type in psfs:
                for sp_type in psfs[ff_type]:
                    line = line+f'{sp_type+'('+ff_type+')':>25s}'

            line = line+"\n"
            f.write(line)

            line = ""
            for ff_type in psfs:
                for sp_type in psfs[ff_type]:
                    line = line+f'{psfs[ff_type][sp_type]:25.13E}'
            line = line+"\n"
            f.write(line)

        f.write("# Spectra:\n")
        line = f"{'E':>10s}"
        for ff_type in spectra:
            for sp_type in spectra[ff_type]:
                line = line+f'{sp_type+'('+ff_type+')':>25s}'

        line = line+"\n"
        f.write(line)

        for ie in range(len(e_grid)):
            line = f"{e_grid[ie]:10.5f}"
            for ff_type in spectra:
                for sp_type in spectra[ff_type]:
                    line = line+f'{spectra[ff_type][sp_type][ie]:25.13E}'
            line = line+"\n"
            f.write(line)


def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    data = {}
    data['PSFs'] = {}
    data['Spectra'] = {}

    headers_psfs = []
    headers_spectra = []
    read_psfs = False
    read_spectra = False
    i_point = 0

    for line in lines:
        if line[0] == '#':
            if "PSFs:" in line:
                data['PSFs'] = {}
                read_psfs = True
                read_spectra = False
                continue
            elif "Spectra:" in line:
                data['Spectra'] = {}
                read_psfs = False
                read_spectra = True
                continue
            elif 'N energy points' in line:
                tokens = line.strip().split("=")
                data["energy_grid"] = np.zeros(int(tokens[1]))
                continue
            else:
                continue

        if read_psfs:
            tokens = line.strip().split()
            if tokens[0][0] == 'G':
                # means we're reading header line
                for tok in tokens:
                    headers_psfs.append(tok)
                    # split at brackets
                    toks = tok.split('(')
                    key_ff = toks[1].strip(')')
                    sp_type = toks[0].strip()
                    if key_ff in data['PSFs']:
                        data['PSFs'][key_ff][sp_type] = 0.
                    else:
                        data['PSFs'][key_ff] = {sp_type: 0.}
            else:
                for i, tok in enumerate(tokens):
                    toks = headers_psfs[i].split('(')
                    key_ff = toks[1].strip(')')
                    sp_type = toks[0].strip()

                    data['PSFs'][key_ff][sp_type] = float(tok)
                read_psfs = False

        if read_spectra:
            tokens = line.strip().split()
            if tokens[0][0] == 'E':
                # prepare arrays
                for i, tok in enumerate(tokens):
                    if i == 0:
                        continue
                    headers_spectra.append(tok)
                    toks = tok.split('(')
                    key_ff = toks[1].strip(')')
                    sp_type = toks[0].strip()
                    if not (key_ff in data['Spectra']):
                        data['Spectra'][key_ff] = {sp_type: np.zeros_like(
                            data["energy_grid"])}
                    else:
                        data['Spectra'][key_ff][sp_type] = np.zeros_like(
                            data["energy_grid"])
            else:
                for i, tok in enumerate(tokens):
                    if i == 0:
                        data["energy_grid"][i_point] = float(tok)
                        continue
                    toks = headers_spectra[i-1].split('(')
                    key_ff = toks[1].strip(')')
                    sp_type = toks[0].strip()

                    data['Spectra'][key_ff][sp_type][i_point] = float(tok)
                i_point += 1

    return data


header_spectra_2D = '''
# PSFs and 2D (e_1, e_2) spectra computed with project_name version
# Energy unit = {energy_unit}
# PSF unit = {psf_unit}
# Rectangular grid: eta1 in [0, q_value-2*emin], eta2 in [0, 1]
# Transformation: e1 = eta1 + emin,
#                 e2 = eta2*(q_value - eta1 - 2*emin) + emin
# N points = {n_eta1} {n_eta2}
# Spectra are normalized to unit integral
# emin = {emin}
# Legend:
# - dG/deta1deta2 = angle-independent electron energy spectrum
# - dH/deta1deta2 = angle-dependent electron energy spectrum
'''


def write_2d_spectra(file_name, eta1_grid: np.ndarray, eta2_grid: np.ndarray, emin: float, spectra: dict, psfs: dict | None = None):
    with open(file_name, "w") as f:
        f.write(header_spectra_2D.format(energy_unit=ph.user_energy_unit_name,
                                         psf_unit=ph.user_psf_unit_name,
                                         n_eta1=len(eta1_grid), n_eta2=len(eta2_grid),
                                         emin=emin))

        if not (psfs is None):
            f.write("# PSFs:\n")
            line = ""
            for ff_type in psfs:
                for sp_type in psfs[ff_type]:
                    line = line+f'{sp_type+'('+ff_type+')':>25s}'

            line = line+"\n"
            f.write(line)

            line = ""
            for ff_type in psfs:
                for sp_type in psfs[ff_type]:
                    line = line+f'{psfs[ff_type][sp_type]:25.13E}'
            line = line+"\n"
            f.write(line)

        f.write("# Spectra:\n")
        line = f"{'E1':>10s} {'E2':>10s}"
        for ff_type in spectra:
            for sp_type in spectra[ff_type]:
                line = line+f'{sp_type+'('+ff_type+')':>25s}'

        line = line+"\n"
        f.write(line)

        for ie in range(len(eta1_grid)):
            e1 = eta1_grid[ie]+emin
            for je in range(len(eta2_grid)):
                e2 = eta2_grid[je]*(eta1_grid[-1]-eta1_grid[ie]) + emin
                # line = f"{e1:10.5f} {e2:10.5f}"
                line = f"{eta1_grid[ie]:10.5f} {eta2_grid[je]:10.5f}"
                for ff_type in spectra:
                    for sp_type in spectra[ff_type]:
                        line = line +\
                            f'{spectra[ff_type][sp_type][ie][je]:25.13E}'
                line = line+"\n"
                f.write(line)


def load_2d_spectra(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    data = {}
    data['PSFs'] = {}
    data['Spectra'] = {}

    headers_psfs = []
    headers_spectra = []
    read_psfs = False
    read_spectra = False
    i_point = 0
    j_point = 0

    for line in lines:
        if line[0] == '#':
            if "PSFs:" in line:
                data['PSFs'] = {}
                read_psfs = True
                read_spectra = False
                continue
            elif "Spectra:" in line:
                data['Spectra'] = {}
                read_psfs = False
                read_spectra = True
                continue
            elif 'N points' in line:
                tokens = line.strip().split("=")
                toks = tokens[1].split()
                data["n_eta_1"] = int(toks[0])
                data["n_eta_2"] = int(toks[1])
                data["e1_grid"] = np.zeros(int(toks[0]))
                data["e2_grid"] = np.zeros(int(toks[1]))
                continue
            elif '# emin' in line:
                tokens = line.strip().split("=")
                data["emin"] = float(tokens[1].strip())
            else:
                continue

        if read_psfs:
            tokens = line.strip().split()
            if tokens[0][0] == 'G':
                # means we're reading header line
                for tok in tokens:
                    headers_psfs.append(tok)
                    # split at brackets
                    toks = tok.split('(')
                    key_ff = toks[1].strip(')')
                    sp_type = toks[0].strip()
                    if key_ff in data['PSFs']:
                        data['PSFs'][key_ff][sp_type] = 0.
                    else:
                        data['PSFs'][key_ff] = {sp_type: 0.}
            else:
                for i, tok in enumerate(tokens):
                    toks = headers_psfs[i].split('(')
                    key_ff = toks[1].strip(')')
                    sp_type = toks[0].strip()

                    data['PSFs'][key_ff][sp_type] = float(tok)
                read_psfs = False

        if read_spectra:
            tokens = line.strip().split()
            if tokens[0][0] == 'E':
                # prepare arrays
                for i, tok in enumerate(tokens):
                    if i < 2:
                        continue
                    headers_spectra.append(tok)
                    toks = tok.split('(')
                    key_ff = toks[1].strip(')')
                    sp_type = toks[0].strip()
                    if not (key_ff in data['Spectra']):
                        data['Spectra'][key_ff] = {sp_type: np.zeros(
                            (len(data['e1_grid']), len(data['e2_grid'])))}
                    else:
                        data['Spectra'][key_ff][sp_type] = np.zeros(
                            (len(data['e1_grid']), len(data['e2_grid'])))
            else:
                for i, tok in enumerate(tokens):
                    if i == 0:
                        if j_point == 0:
                            data['e1_grid'][i_point] = float(tok)

                        continue
                    if i == 1:
                        if i_point == 0:
                            data['e2_grid'][j_point] = float(tok)
                        continue

                    toks = headers_spectra[i-2].split('(')
                    key_ff = toks[1].strip(')')
                    sp_type = toks[0].strip()

                    data['Spectra'][key_ff][sp_type][i_point][j_point] = float(
                        tok)
                j_point += 1
                if j_point == data["n_eta_2"]:
                    j_point = 0
                    i_point += 1
    return data
