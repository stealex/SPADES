import re
import numpy as np
from . import ph, radial_wrapper, math_stuff, dhfs
from .math_stuff import hydrogenic_binding_energy, coulomb_phase_shift
from .radial_wrapper import RADIALError
from .dhfs import dhfs_handler
from tqdm import tqdm


class bound_config:
    def __init__(self, max_r: float, n_radial_points: int,
                 n_values: int | tuple[int, int] | list[int],
                 k_values: str | int | tuple[int, int] | dict[int, list[int]] = "auto") -> None:
        self.max_r = max_r
        self.n_radial_points = n_radial_points

        if (type(n_values) == int):
            self.n_values = [n_values]
        elif (type(n_values) == tuple):
            self.n_values = range(n_values[0], n_values[1]+1)
        elif (type(n_values) == list):
            self.n_values = n_values

        self.k_values = {}
        for i_n in range(len(self.n_values)):
            n = self.n_values[i_n]
            if (type(k_values) == str):
                if k_values != "auto":
                    raise ValueError("Cannot interpret k_values option")
                else:
                    k_tmp = range(-n, n)
            if (type(k_values) == int):
                if (k_values < -n) or (k_values >= n) or (k_values == 0):
                    continue
                self.k_values[n] = k_values
                continue
            elif type(k_values) == tuple:
                k_tmp = range(k_values[0], k_values[1]+1)
            elif type(k_values) == dict:
                self.k_values = k_values
                break

            for i_k in range(len(k_tmp)):
                if (k_tmp[i_k] < -n) or (k_tmp[i_k] >= n) or (k_tmp[i_k] == 0):
                    continue
                if (n not in self.k_values):
                    self.k_values[n] = []
                self.k_values[n].append(k_tmp[i_k])

    def print(self):
        print("Configuration for bound states")
        print(f"  - Maximum radial distance: {self.max_r*ph.to_distance_units(): 8.3f}",
              f"{ph.user_distance_unit_name}")
        print(f"  - Number of radial points: {self.n_radial_points: d}")
        print(f"  - N and K values:")
        for i in range(len(self.n_values)):
            print(f"    - {self.n_values[i]}", self.k_values[self.n_values[i]])


class bound_handler:
    def __init__(self, z_nuc: int, n_e: int, bound_states_configuration: bound_config) -> None:
        self.config = bound_states_configuration
        self.z_nuc = z_nuc
        self.n_e = n_e

        _, r, dr = radial_wrapper.call_sgrid(self.config.max_r*ph.fm/ph.bohr_radius,
                                             1E-7,
                                             0.5,
                                             self.config.n_radial_points,
                                             2*self.config.n_radial_points)
        self.r_grid = r*ph.bohr_radius/ph.fm
        self.dr_grid = dr*ph.bohr_radius/ph.fm

        radial_wrapper.call_setrgrid(r)

    def set_potential(self, r_grid: np.ndarray, rv_grid: np.ndarray):
        radial_wrapper.call_vint(
            r_grid*ph.fm/ph.bohr_radius, rv_grid*ph.MeV*ph.fm/(ph.hartree_energy*ph.bohr_radius))

    def find_bound_states(self, binding_energies=None):
        self.p_grid = {}
        self.q_grid = {}
        self.be = {}

        for i_n in tqdm(range(len(self.config.n_values)),
                        desc="Computing bound states",
                        ncols=100):
            n = self.config.n_values[i_n]
            self.p_grid[n] = {}
            self.q_grid[n] = {}
            self.be[n] = {}

            for k in tqdm(self.config.k_values[n],
                          desc=f"n={n:d}",
                          ncols=100,
                          leave=False):
                if not (binding_energies is None):
                    trial_be = binding_energies[n][k]
                else:
                    trial_be = math_stuff.hydrogenic_binding_energy(
                        self.z_nuc, n, k)
                try:
                    true_be = radial_wrapper.call_dbound(
                        trial_be/ph.hartree_energy, n, k)
                except RADIALError:
                    # means computation did not succeed for whatever reason
                    # don't stop, just don't add this to the class
                    print(f"could not find bound state for {n:d}, {k:d}")
                    continue

                p, q = radial_wrapper.call_getpq(
                    self.config.n_radial_points)
                self.p_grid[n][k] = p * 1./np.sqrt(ph.bohr_radius/ph.fm)
                self.q_grid[n][k] = q * 1./np.sqrt(ph.bohr_radius/ph.fm)
                self.be[n][k] = true_be*ph.hartree_energy


class scattering_config:
    def __init__(self, max_r: float, n_radial_points: int,
                 min_ke: float, max_ke: float, n_ke_points: int,
                 k_values: int | tuple[int, int] | list[int]) -> None:
        self.max_r = max_r
        self.n_radial_points = n_radial_points

        self.min_ke = min_ke
        self.max_ke = max_ke
        self.n_ke_points = n_ke_points

        self.k_values = []
        if type(k_values) == int:
            self.k_values.append(k_values)
        elif type(k_values) == tuple:
            for i_k in range(k_values[0], k_values[1]+1):
                if (k_values[i_k] == 0):
                    continue
                self.k_values.append(k_values[i_k])
        elif type(k_values) == list:
            for k in k_values:
                if (k == 0):
                    continue
                self.k_values.append(k)
        else:
            raise ValueError("Cannot interpret the type of k_values")

    def print(self):
        print("Configuration for scattering states")
        print(f"  - Maximum radial distance: {self.max_r*ph.to_distance_units(): 8.3f}"
              f" {ph.user_distance_unit_name}")
        print(f"  - Number of radial points: {self.n_radial_points: d}")
        print(f"  - K values: ", self.k_values)
        print(f"  - Energy limits: [", self.min_ke/ph.user_energy_unit,
              ", ", self.max_ke/ph.user_energy_unit, f"] {ph.user_energy_unit_name}")
        print(f"  - Number of energy points: {self.n_ke_points}")


class scattering_handler:
    def __init__(self, z_nuc: int, n_e: int, scattering_states_configuration: scattering_config) -> None:
        self.config = scattering_states_configuration

        _, r, dr = radial_wrapper.call_sgrid(self.config.max_r*ph.fm/ph.bohr_radius,
                                             1E-7,
                                             0.5,
                                             self.config.n_radial_points,
                                             2*self.config.n_radial_points)

        self.r_grid = r*ph.bohr_radius/ph.fm
        self.dr_grid = dr*ph.bohr_radius/ph.fm

        radial_wrapper.call_setrgrid(r)
        self.energy_grid = np.logspace(np.log10(self.config.min_ke),
                                       np.log10(self.config.max_ke),
                                       self.config.n_ke_points)

    def set_potential(self, r_grid: np.ndarray, rv_grid: np.ndarray):
        to_atomic = ph.MeV*ph.fm/(ph.hartree_energy*ph.bohr_radius)
        self.z_inf = rv_grid[-1]*to_atomic
        radial_wrapper.call_vint(
            r_grid*ph.fm/ph.bohr_radius, rv_grid*to_atomic)

    def compute_scattering_states(self):
        self.phase_grid = {}
        self.coul_phase_grid = {}
        self.p_grid = {}
        self.q_grid = {}
        self.norm = np.sqrt((self.energy_grid+2.0*ph.electron_mass) /
                            (2.0*(self.energy_grid+ph.electron_mass)))

        for i_k in tqdm(range(len(self.config.k_values)),
                        desc="Computing scattering states",
                        ncols=100):
            k = self.config.k_values[i_k]
            self.phase_grid[k] = np.zeros_like(self.energy_grid)
            self.coul_phase_grid[k] = np.zeros_like(self.energy_grid)
            self.p_grid[k] = np.zeros(
                (len(self.energy_grid), len(self.r_grid)))
            self.q_grid[k] = np.zeros_like(self.p_grid[k])

            for i_e in tqdm(range(len(self.energy_grid)),
                            desc=f"k={k:d}",
                            ncols=100,
                            leave=False):
                e = self.energy_grid[i_e]
                try:
                    phase = radial_wrapper.call_dfree(
                        e/ph.hartree_energy, k, 1E-14)
                except RADIALError:
                    print(f"Could not find scattering state {k:d}, {e:f}")
                    continue

                p, q = radial_wrapper.call_getpq(len(self.r_grid))
                coul_phase_shift = coulomb_phase_shift(
                    e, self.z_inf, k)
                self.p_grid[k][i_e] = p
                self.q_grid[k][i_e] = q

                self.phase_grid[k][i_e] = phase  # + coul_phase_shift
                self.coul_phase_grid[k][i_e] = coul_phase_shift


class wavefunctions_handler:
    def __init__(self, atom: dhfs.atomic_system, bound_conf: bound_config | None = None, scattering_conf: scattering_config | None = None) -> None:
        self.atomic_system = atom
        if (bound_conf is None) and (scattering_conf is None):
            raise ValueError(
                "At least one of bound_conf or scattering_conf should be passed")
        if not (bound_conf is None):
            self.bound_config = bound_conf
        if not (scattering_conf is None):
            self.scattering_config = scattering_conf

    def run_dhfs(self) -> None:
        self.dhfs_handler = dhfs_handler(
            self.atomic_system, self.atomic_system.name)
        self.dhfs_handler.run_dhfs(self.bound_config.max_r,
                                   self.bound_config.n_radial_points)

        self.dhfs_handler.retrieve_dhfs_results()
        self.dhfs_handler.build_modified_potential()

    def find_bound_states(self):
        self.bound_handler = bound_handler(self.atomic_system.Z,
                                           int(self.atomic_system.occ_values.sum()),
                                           self.bound_config)
        self.bound_handler.set_potential(self.dhfs_handler.rad_grid,
                                         self.dhfs_handler.rv_modified)
        self.bound_handler.find_bound_states()

    def find_scattering_states(self, r_grid_scattering: np.ndarray | None = None, rv_scattering: np.ndarray | None = None):
        # solve scattering states in final atom
        self.scattering_handler = scattering_handler(self.atomic_system.Z,
                                                     int(self.atomic_system.occ_values.sum(
                                                     )),
                                                     self.scattering_config)
        if (rv_scattering is None) and (r_grid_scattering is None):
            print("Will use dhfs potential for scattering states")
            self.scattering_handler.set_potential(
                self.dhfs_handler.rad_grid, self.dhfs_handler.rv_modified)
        elif (not (rv_scattering is None)) and (not (r_grid_scattering is None)):
            print("Will use user potential for scattering states")
            self.scattering_handler.set_potential(
                r_grid_scattering, rv_scattering)

        self.scattering_handler.compute_scattering_states()

    def find_all_wavefunctions(self, r_grid_scattering: np.ndarray | None = None, rv_scattering: np.ndarray | None = None):
        try:
            self.run_dhfs()
            self.find_bound_states()
        except AttributeError:
            pass

        try:
            self.find_scattering_states(r_grid_scattering, rv_scattering)
        except AttributeError:
            pass
