"""Bound and scattering electron wavefunction solvers."""
from __future__ import annotations

import numpy as np

from spades.config import BoundConfig, ScatteringConfig
from . import ph, radial_wrapper, math_stuff, dhfs
from spades.math_stuff import coulomb_phase_shift
from .radial_wrapper import RADIALError
from .dhfs import AtomicSystem, DHFSHandler, create_ion
from tqdm import tqdm
from scipy.interpolate import CubicSpline

import logging
logger = logging.getLogger(__name__)


class BoundHandler:
    """Compute bound-state Dirac wavefunctions for a fixed central potential."""

    def __init__(self, z_nuc: int, n_e: int, bound_states_configuration: BoundConfig) -> None:
        """Build radial grid and initialize RADIAL for bound-state solutions.

        Parameters
        ----------
        z_nuc:
            Nuclear charge.
        n_e:
            Number of electrons (reserved for future use).
        bound_states_configuration:
            Bound-state configuration object.
        """
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
        """Load ``r*V(r)`` potential into the RADIAL backend.

        Parameters
        ----------
        r_grid:
            Radial grid in fm.
        rv_grid:
            ``r * V(r)`` values in MeV*fm.
        """
        radial_wrapper.call_vint(
            r_grid*ph.fm/ph.bohr_radius, rv_grid*ph.MeV*ph.fm/(ph.hartree_energy*ph.bohr_radius))

    def find_bound_states(self, binding_energies=None):
        """Solve requested bound shells and build interpolation splines.

        Parameters
        ----------
        binding_energies:
            Optional nested dictionary of trial binding energies keyed by ``n`` and ``kappa``.
        """
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
                    logger.info(f"Testing {trial_be/ph.hartree_energy}")
                    true_be = radial_wrapper.call_dbound(
                        trial_be/ph.hartree_energy, n, k)
                    logger.debug("Obtained bound-state energy %s for n=%d, k=%d", true_be, n, k)
                except RADIALError:
                    # means computation did not succeed for whatever reason
                    # don't stop, just don't add this to the class
                    logger.warning(
                        "Could not find bound state for n=%d, k=%d", n, k)
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
        """Evaluate shell probability density at radius ``radius``.

        Parameters
        ----------
        radius:
            Radius in fm where the density is evaluated.
        n:
            Principal quantum number.
        kappa:
            Relativistic angular quantum number.

        Returns
        -------
        float
            Probability-density-like factor used by capture-channel spectra.
        """
        constant_term = 1.0/(4.0*np.pi*(ph.electron_mass**3.0)) *\
            (ph.hc**3.0)
        function_term = (self.p_func[n][kappa](radius)**2.0 +
                         self.q_func[n][kappa](radius)**2.0)/(radius**2.0)
        return constant_term*function_term


class ScatteringHandler:
    """Compute continuum Dirac wavefunctions and phase shifts."""

    def __init__(self, z_nuc: int, n_e: int, scattering_states_configuration: ScatteringConfig) -> None:
        """Build radial and kinetic-energy grids for scattering states.

        Parameters
        ----------
        z_nuc:
            Nuclear charge.
        n_e:
            Number of electrons (reserved for future use).
        scattering_states_configuration:
            Scattering-state configuration object.
        """
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
        """Load potential and extract asymptotic charge for Coulomb phase shifts.

        Parameters
        ----------
        r_grid:
            Radial grid in fm.
        rv_grid:
            ``r * V(r)`` values in MeV*fm.
        """
        to_atomic = ph.MeV*ph.fm/(ph.hartree_energy*ph.bohr_radius)
        self.z_inf = rv_grid[-1]*to_atomic
        radial_wrapper.call_vint(
            r_grid*ph.fm/ph.bohr_radius, rv_grid*to_atomic)

    def compute_scattering_states(self):
        """Solve the continuum equation for all configured ``kappa`` and energies."""
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
                    logger.warning(
                        "Could not find scattering state k=%d, E=%f", k, e)
                    continue

                p, q = radial_wrapper.call_getpq(len(self.r_grid))
                coul_phase_shift = coulomb_phase_shift(
                    e, self.z_inf, k)
                self.p_grid[k][i_e] = p
                self.q_grid[k][i_e] = q

                self.phase_grid[k][i_e] = phase  # coul_phase_shift
                self.coul_phase_grid[k][i_e] = coul_phase_shift


class WaveFunctionsHandler:
    """High-level orchestrator for DHFS, bound, and scattering wavefunctions."""

    def __init__(self, atom: dhfs.AtomicSystem, bound_conf: BoundConfig | None = None, scattering_conf: ScatteringConfig | None = None, rad_grid: np.ndarray | None = None, rv_grid: np.ndarray | None = None) -> None:
        """Create a wavefunction workflow for one atomic system.

        Parameters
        ----------
        atom:
            Atomic system for which wavefunctions are computed.
        bound_conf:
            Optional bound-state configuration.
        scattering_conf:
            Optional scattering-state configuration.
        rad_grid, rv_grid:
            Optional precomputed potential grid and ``r*V(r)`` values in fm/MeV*fm.
        """
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
        """Run DHFS workflow for neutral or positively charged ions."""
        self.dhfs_handler = DHFSHandler(
            self.atomic_system, self.atomic_system.name)
        self.dhfs_handler.run_dhfs(self.bound_config.max_r,
                                   self.bound_config.n_radial_points)

        self.dhfs_handler.retrieve_dhfs_results()
        self.dhfs_handler.build_modified_potential()

        self.rad_grid = self.dhfs_handler.rad_grid
        self.rv_grid = self.dhfs_handler.rv_modified

    def run_dhfs_negative_ion(self):
        """Construct a modified potential for negative ions from reference systems."""
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
        """Dispatch DHFS handling based on net ionic charge."""
        if (self.atomic_system.net_charge() >= 0):
            self.run_dhfs_neutral_or_positive_ion()
        else:
            self.run_dhfs_negative_ion()

    def find_bound_states(self):
        """Solve bound states if the process requires them."""
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
        """Solve scattering states on the current potential.

        Parameters
        ----------
        r_grid_scattering, rv_scattering:
            Reserved optional arguments for alternative scattering potentials.
        """
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
        """Run the complete bound/scattering workflow with DHFS fallback."""
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
