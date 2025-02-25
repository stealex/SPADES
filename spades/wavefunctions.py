import re
from hepunits import rad
import numpy as np

from spades.config import BoundConfig, ScatteringConfig
from . import ph, radial_wrapper, math_stuff, dhfs
from spades.math_stuff import hydrogenic_binding_energy, coulomb_phase_shift
from .radial_wrapper import RADIALError
from .dhfs import AtomicSystem, DHFSHandler, create_ion
from tqdm import tqdm
from scipy.interpolate import CubicSpline

import logging
logger = logging.getLogger(__name__)


# class bound_config_old:
#     def __init__(self, max_r: float, n_radial_points: int,
#                  n_values: int | tuple[int, int] | list[int],
#                  k_values: str | int | tuple[int, int] | dict[int, list[int]] = "auto") -> None:
#         self.max_r = max_r
#         self.n_radial_points = n_radial_points

#         if (type(n_values) == int):
#             self.n_values = [n_values]
#         elif (type(n_values) == tuple):
#             self.n_values = range(n_values[0], n_values[1]+1)
#         elif (type(n_values) == list):
#             self.n_values = n_values

#         self.k_values = {}
#         for i_n in range(len(self.n_values)):
#             n = self.n_values[i_n]
#             if (type(k_values) == str):
#                 if k_values != "auto":
#                     raise ValueError("Cannot interpret k_values option")
#                 else:
#                     k_tmp = range(-n, n)
#             if (type(k_values) == int):
#                 if (k_values < -n) or (k_values >= n) or (k_values == 0):
#                     continue
#                 self.k_values[n] = k_values
#                 continue
#             elif type(k_values) == tuple:
#                 k_tmp = range(k_values[0], k_values[1]+1)
#             elif type(k_values) == dict:
#                 self.k_values = k_values
#                 break

#             for i_k in range(len(k_tmp)):
#                 if (k_tmp[i_k] < -n) or (k_tmp[i_k] >= n) or (k_tmp[i_k] == 0):
#                     continue
#                 if (n not in self.k_values):
#                     self.k_values[n] = []
#                 self.k_values[n].append(k_tmp[i_k])

#     def print(self):
#         print("Configuration for bound states")
#         print(f"  - Maximum radial distance: {self.max_r*ph.to_distance_units(): 8.3f}",
#               f"{ph.user_distance_unit_name}")
#         print(f"  - Number of radial points: {self.n_radial_points: d}")
#         print(f"  - N and K values:")
#         for i in range(len(self.n_values)):
#             print(f"    - {self.n_values[i]}", self.k_values[self.n_values[i]])


class BoundHandler:
    def __init__(self, z_nuc: int, n_e: int, bound_states_configuration: BoundConfig) -> None:
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
        self.p_func = {}
        self.q_func = {}
        self.be = {}

        for i_n in tqdm(range(len(self.config.n_values)),
                        desc="Computing bound states",
                        ncols=100):
            n = self.config.n_values[i_n]
            self.p_grid[n] = {}
            self.q_grid[n] = {}
            self.p_func[n] = {}
            self.q_func[n] = {}
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

                self.p_func[n][k] = CubicSpline(
                    self.r_grid,
                    self.p_grid[n][k]
                )

                self.q_func[n][k] = CubicSpline(
                    self.r_grid,
                    self.q_grid[n][k]
                )

    def probability_in_sphere(self, radius: float, n: int, kappa: int):
        constant_term = 1.0/(4.0*np.pi*(ph.electron_mass**3.0)) *\
            (ph.hc**3.0)
        function_term = (self.p_func[n][kappa](radius)**2.0 +
                         self.q_func[n][kappa](radius)**2.0)/(radius**2.0)
        return constant_term*function_term
# class scattering_config_old:
#     def __init__(self, max_r: float, n_radial_points: int,
#                  min_ke: float, max_ke: float, n_ke_points: int,
#                  k_values: int | tuple[int, int] | list[int]) -> None:
#         self.max_r = max_r
#         self.n_radial_points = n_radial_points

#         self.min_ke = min_ke
#         self.max_ke = max_ke
#         self.n_ke_points = n_ke_points

#         self.k_values = []
#         if type(k_values) == int:
#             self.k_values.append(k_values)
#         elif type(k_values) == tuple:
#             for i_k in range(k_values[0], k_values[1]+1):
#                 if (k_values[i_k] == 0):
#                     continue
#                 self.k_values.append(k_values[i_k])
#         elif type(k_values) == list:
#             for k in k_values:
#                 if (k == 0):
#                     continue
#                 self.k_values.append(k)
#         else:
#             raise ValueError("Cannot interpret the type of k_values")

#     def print(self):
#         print("Configuration for scattering states")
#         print(f"  - Maximum radial distance: {self.max_r*ph.to_distance_units(): 8.3f}"
#               f" {ph.user_distance_unit_name}")
#         print(f"  - Number of radial points: {self.n_radial_points: d}")
#         print(f"  - K values: ", self.k_values)
#         print(f"  - Energy limits: [", self.min_ke/ph.user_energy_unit,
#               ", ", self.max_ke/ph.user_energy_unit, f"] {ph.user_energy_unit_name}")
#         print(f"  - Number of energy points: {self.n_ke_points}")


class ScatteringHandler:
    def __init__(self, z_nuc: int, n_e: int, scattering_states_configuration: ScatteringConfig) -> None:
        self.config = scattering_states_configuration

        logger.debug(
            f"creating r grid with r_max={self.config.max_r*ph.fm/ph.bohr_radius}, N = {self.config.n_radial_points}")
        _, r, dr = radial_wrapper.call_sgrid(self.config.max_r*ph.fm/ph.bohr_radius,
                                             1E-7,
                                             0.5,
                                             self.config.n_radial_points,
                                             2*self.config.n_radial_points)
        self.r_grid = r*ph.bohr_radius/ph.fm
        self.dr_grid = dr*ph.bohr_radius/ph.fm

        radial_wrapper.call_setrgrid(r)

        logger.debug(
            f"creating energy grid with E_min={self.config.min_ke}, E_max={self.config.max_ke}, N={self.config.n_ke_points}")
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
        self.delta_infinity = radial_wrapper.call_delinf()

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

            delta_corr = self.phase_grid[k].copy()
            p_corr = self.p_grid[k].copy()
            q_corr = self.q_grid[k].copy()
            for i in range(len(delta_corr)-1, 0, -1):
                if (np.abs(delta_corr[i] - delta_corr[i-1]) > 1.0):
                    # print("found discontinuity at ", i)
                    delta_corr[:i] = delta_corr[:i] + np.pi
                    p_corr[:i] = -1.*p_corr[:i]
                    q_corr[:i] = -1.*q_corr[:i]

            self.phase_grid[k] = delta_corr
            self.p_grid[k] = p_corr
            self.q_grid[k] = q_corr


class WaveFunctionsHandler:
    def __init__(self, atom: dhfs.AtomicSystem, bound_conf: BoundConfig | None = None, scattering_conf: ScatteringConfig | None = None, rad_grid: np.ndarray | None = None, rv_grid: np.ndarray | None = None) -> None:
        self.atomic_system = atom
        if (bound_conf is None) and (scattering_conf is None):
            raise ValueError(
                "At least one of bound_conf or scattering_conf should be passed")
        if not (bound_conf is None):
            self.bound_config = bound_conf
        if not (scattering_conf is None):
            self.scattering_config = scattering_conf

        self.rad_grid = None
        self.rv_grid = None
        if not ((rad_grid is None) or (rv_grid is None)):
            self.rad_grid = rad_grid
            self.rv_grid = rv_grid

    def run_dhfs_neutral_or_positive_ion(self):
        self.dhfs_handler = DHFSHandler(
            self.atomic_system, self.atomic_system.name)
        self.dhfs_handler.run_dhfs(self.bound_config.max_r,
                                   self.bound_config.n_radial_points)

        self.dhfs_handler.retrieve_dhfs_results()
        self.dhfs_handler.build_modified_potential()

        self.rad_grid = self.dhfs_handler.rad_grid
        self.rv_grid = self.dhfs_handler.rv_modified

    def run_dhfs_negative_ion(self):
        z_ref = self.atomic_system.Z

        # create the neutral version of this atom
        # and run DHFS for it
        neutral_atom_ref = AtomicSystem(
            atomic_number=z_ref, mass_number=self.atomic_system.mass_number)
        handler_neutral_ref = DHFSHandler(neutral_atom_ref, "")
        handler_neutral_ref.run_dhfs(self.bound_config.max_r,
                                     self.bound_config.n_radial_points)
        handler_neutral_ref.retrieve_dhfs_results()

        # create positive ion version of our atom
        # first create the neutral atom of less charge
        neutral_atom_zminus = AtomicSystem(
            atomic_number=z_ref+self.atomic_system.net_charge(),
            mass_number=self.atomic_system.mass_number+self.atomic_system.net_charge())
        positive_ion = create_ion(
            neutral_atom_zminus, z_ref)  # adds 1 or 2 protons
        handler_positive_ion = DHFSHandler(positive_ion, "")
        handler_positive_ion.run_dhfs(self.bound_config.max_r,
                                      self.bound_config.n_radial_points)
        handler_positive_ion.retrieve_dhfs_results()

        # the reference, neutral and positive ions have the same number of protons
        # so the electrostatic potential of the nucleus will be almost the same
        rv = handler_neutral_ref.rv_nuc
        delta_pot = handler_neutral_ref.rv_el - handler_positive_ion.rv_el
        rv = rv + handler_neutral_ref.rv_el+delta_pot

        self.rad_grid = handler_neutral_ref.rad_grid
        self.rv_grid = -rv

    def run_dhfs(self) -> None:
        if (self.atomic_system.net_charge() >= 0):
            self.run_dhfs_neutral_or_positive_ion()
        else:
            self.run_dhfs_negative_ion()

    def find_bound_states(self):
        if (self.atomic_system.net_charge() < 0):
            # we're dealing with a betaPlus mode.
            # no need to run bound states for it
            return
        self.bound_handler = BoundHandler(self.atomic_system.Z,
                                          int(self.atomic_system.occ_values.sum()),
                                          self.bound_config)
        if (self.rad_grid is None) or (self.rv_grid is None):
            raise ValueError("Internal inconsistency")

        self.bound_handler.set_potential(self.rad_grid,
                                         self.rv_grid)
        self.bound_handler.find_bound_states()

    def find_scattering_states(self, r_grid_scattering: np.ndarray | None = None, rv_scattering: np.ndarray | None = None):
        # solve scattering states in final atom
        self.scattering_handler = ScatteringHandler(self.atomic_system.Z,
                                                    int(self.atomic_system.occ_values.sum(
                                                    )),
                                                    self.scattering_config)
        if (self.rad_grid is None) or (self.rv_grid is None):
            raise ValueError("Internal inconsistency")
        self.scattering_handler.set_potential(self.rad_grid,
                                              self.rv_grid)
        self.scattering_handler.compute_scattering_states()

    def find_all_wavefunctions(self):
        if self.rv_grid is None:
            # we did not receive a custom potential. Will use dhfs to compute it

            try:
                self.run_dhfs()
                self.find_bound_states()
            except AttributeError:
                pass

            try:
                self.find_scattering_states()
            except AttributeError:
                pass
        else:
            # we received a custom potential. Solve bound/scattering states in it
            try:
                self.find_bound_states()
            except AttributeError:
                pass

            try:
                self.find_scattering_states()
            except AttributeError:
                pass
