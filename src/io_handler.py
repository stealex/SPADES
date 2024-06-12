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
