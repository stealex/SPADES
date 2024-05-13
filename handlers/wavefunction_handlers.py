from wrappers import dhfs_wrapper, radial_wrapper
from wrappers.radial_wrapper import RADIALError
import periodictable
from utils.math_stuff import hydrogenic_binding_energy, coulomb_phase_shift
from utils import ph
import numpy as np
from configs.wavefunctions_config import atomic_system, radial_bound_config, radial_scattering_config


class dhfs_handler:
    """
    Class used for handling DHFS calculations and results for a specific system.
    Example usage:
        from utils import dhfs_handler

        handler = dhfs_handler(config, label)
        handler.run_dhfs(100, radius_unit='bohr')
        handler.rertrieve_results()

    Attributes
    ----------
    label : str
        label for bookkeeping and printing
    dhfs_config: dhfs_configuration
        electron configuration and nuclear charge. See documentation of class dhfs_configuration
    atomic_weight: float
        mass number in g/mol
    rad_grid: np.ndarray
        radial grid where wavefunction are computed
    p_grid: np.ndarray
        values of the P component of the wavefunctions. Dimensions = (nb shells, nb radial points)
    q_grid: np.ndarray
        values of the Q component of the wavefunctions. Dimensions = (nb shells, nb radial points)
    rv_nuc: np.ndarray
        values of the r times electrostatic potential of the nucleus. Dimensions = len(rad_grid)
    rv_el: np.ndarray
        values of the r times electronic potential. Dimensions = len(rad_grid)
    rv_ex: np.ndarray
        values of the r times exchange potential. Dimensions = len(rad_grid)
    """

    def __init__(self, config: dict | atomic_system, label: str, atomic_weight: float = -1.0) -> None:
        """Handler class for DHFS.f usage.

        Args:
            configuration_file (dict): dictonary {"name", "label, "electron_config"}
            label (str): Relevant label for calculations.
            atomic_weight (float, optional): Atomic weight in g/mol. If negative, the value from periodictable is used. Defaults to -1
        """
        self.label = label
        if (type(config) == dict):
            self.dhfs_config = atomic_system(config)
        elif (type(config) == atomic_system):
            self.dhfs_config = config
        else:
            raise TypeError(
                "config should be either a dictionary or an object of type atomic_system")

        if atomic_weight > 0.:
            self.atomic_weight = atomic_weight
        else:
            self.atomic_weight = periodictable.elements[self.dhfs_config.Z].mass

    def print(self):
        print(f"DHFS handler for {self.label}")
        print(f"Atomic weight = {self.atomic_weight:f}")
        self.dhfs_config.print()

    def run_dhfs(self, max_radius: float, n_points=1000, iverbose=0):
        """Organizes the call to DHFS_MAIN from DHFS.f.
        Sets appropriate parameters first.


        Args:
            max_radius (float): Maximum radius for radial grid
            n_points (int, optional): Number of radial points to use (tentative). Defaults to 1000.

        Raises:
            ValueError: _description_
        """

        dhfs_wrapper.call_configuration_input(self.dhfs_config.n_values,
                                              self.dhfs_config.l_values,
                                              self.dhfs_config.jj_values,
                                              self.dhfs_config.occ_values,
                                              self.dhfs_config.Z)

        n_tmp = dhfs_wrapper.call_set_parameters(
            self.atomic_weight, max_radius, n_points, iverbose)
        self.n_grid_points = n_tmp.value

        dhfs_wrapper.call_dhfs_main(1.5, iverbose)

    def retrieve_dhfs_results(self):
        """
    Retrieves results from the DHFS (Dirac-Hartree-Fock-Slater) calculations.

    Calls the DHFS wrapper to obtain wavefunction components (r, p, q) and potential components (vn, vel, vex).

    The wavefunction components are computed based on the electron configuration and grid points specified 
    in dhfs_config and n_grid_points respectively.

    The radial grid 'r' is stored in self.rad_grid, wavefunction components 'p' and 'q' are stored in
    self.p_grid and self.q_grid respectively.

    The potential components 'vn', 'vel', and 'vex' are obtained and stored in self.rv_nuc, self.rv_el, 
    and self.rv_ex respectively.
    """
        (r, p, q) = dhfs_wrapper.call_get_wavefunctions(
            len(self.dhfs_config.electron_config), self.n_grid_points)
        self.rad_grid = r
        self.p_grid = p
        self.q_grid = q

        vn, vel, vex = dhfs_wrapper.call_get_potentials(self.n_grid_points)
        self.rv_nuc = vn
        self.rv_el = vel
        self.rv_ex = vex

        self.binding_energies = dhfs_wrapper.call_get_binding_energies(len(p))

    def build_modified_potential(self) -> np.ndarray:
        """Builds the modified DHFS potential a la [Nitescu et al, Phys. Rev. C 107, 025501, 2023]
        The resulting potential is suitable for the computation of both bound and scattering states
        and yields scattering states orthogonal on bound ones.

        Returns:
            np.ndarray: r times the modified potential
        """
        rv_exchange_modified = np.zeros(len(self.rv_ex))
        density = np.zeros(len(self.rv_el))
        for i_s in range(len(self.dhfs_config.occ_values)):
            p = self.p_grid[i_s]
            q = self.q_grid[i_s]
            occ = self.dhfs_config.occ_values[i_s]
            density = density+occ*(p*p+q*q)

        cslate = 0.75/(np.pi*np.pi)
        rv_exchange_modified = -1.5 * \
            np.power(cslate*self.rad_grid*density, 1./3.)
        self.rv_modified = self.rv_nuc+self.rv_el + rv_exchange_modified
        self.rv_modified[0] = 0.

        return self.rv_modified


class bound_handler:
    def __init__(self, z_nuc: int, n_e: int, bound_states_configuration: radial_bound_config) -> None:
        self.bound_config = bound_states_configuration
        self.z_nuc = z_nuc
        self.n_e = n_e
        self.r_grid = None
        self.rv_grid = None

    def find_bound_states(self, r_grid: np.ndarray, rv_grid: np.ndarray, binding_energies=None):
        self.states = {}
        self.r_grid = r_grid
        self.rv_grid = rv_grid

        radial_wrapper.call_setrgrid(self.r_grid)
        radial_wrapper.call_vint(self.r_grid, self.rv_grid)

        for i_n in range(len(self.bound_config.n_values)):
            n = self.bound_config.n_values[i_n]
            self.states[n] = {}

            for k in self.bound_config.k_values[n]:
                self.states[n][k] = {}

                if not (binding_energies is None):
                    trial_be = binding_energies[n][k]
                else:
                    trial_be = hydrogenic_binding_energy(self.z_nuc, n, k)
                try:
                    true_be = radial_wrapper.call_dbound(trial_be, n, k)
                except RADIALError:
                    # means computation did not succeed for whatever reason
                    # don't stop, just don't add this to the class
                    print(f"could not find bound state for {n:d}, {k:d}")
                    continue

                p, q = radial_wrapper.call_getpq(
                    self.bound_config.n_radial_points)

                self.states[n][k] = {"be": true_be, "p": p, "q": q}


class scattering_handler:
    def __init__(self, z_nuc: int, n_e: int, scattering_states_configuration: radial_scattering_config) -> None:
        self.scattering_config = scattering_states_configuration
        self.z_nuc = z_nuc
        self.n_e = n_e

    def set_potential(self, r_grid: np.ndarray, rv_grid: np.ndarray):
        self.z_inf = rv_grid[-1]
        radial_wrapper.call_vint(r_grid, rv_grid)

    def compute_scattering_states(self):
        self.energy_grid = np.logspace(np.log10(self.scattering_config.min_ke),
                                       np.log10(
                                           self.scattering_config.max_ke),
                                       self.scattering_config.n_ke_points)
        try:
            self.r_grid
        except AttributeError:
            self.scattering_config.n_radial_points, self.r_grid, _ = radial_wrapper.call_sgrid(self.scattering_config.max_r,
                                                                                               1E-7,
                                                                                               0.5,
                                                                                               self.scattering_config.n_radial_points,
                                                                                               5*self.scattering_config.n_radial_points)

        radial_wrapper.call_setrgrid(self.r_grid)
        self.phase_grid = np.zeros(
            (len(self.scattering_config.k_values), len(self.energy_grid)))
        self.p_grid = np.zeros(
            (len(self.scattering_config.k_values), len(self.energy_grid), len(self.r_grid)))
        self.q_grid = np.zeros_like(self.p_grid)
        self.f_grid = np.zeros_like(self.p_grid)
        self.g_grid = np.zeros_like(self.p_grid)

        for i_k in range(len(self.scattering_config.k_values)):
            k = self.scattering_config.k_values[i_k]
            for i_e in range(len(self.energy_grid)):
                e = self.energy_grid[i_e]
                try:
                    phase = radial_wrapper.call_dfree(e, k, 1E-14)
                except RADIALError:
                    print(f"Could not find scattering state {k:d}, {e:f}")
                    continue

                p, q = radial_wrapper.call_getpq(len(self.r_grid))
                self.p_grid[i_k][i_e] = p
                self.q_grid[i_k][i_e] = q

                coul_phase_shift = coulomb_phase_shift(
                    e*ph.hartree_energy, self.z_inf, k)
                self.phase_grid[i_k][i_e] = phase  # + coul_phase_shift
                # print(k, self.z_inf, e*ph.hartree_energy, coul_phase_shift)
                momentum = np.sqrt(e*(e+2*ph.electron_mass/ph.hartree_energy))
                norm = np.sqrt((e + 2.0*ph.electron_mass/ph.hartree_energy) /
                               (2.0*(e+ph.electron_mass/ph.hartree_energy)))

                g = 1.0/ph.fine_structure*norm*1./momentum * p/self.r_grid
                f = 1.0/ph.fine_structure*norm*1./momentum * q/self.r_grid

                self.g_grid[i_k][i_e] = g
                self.f_grid[i_k][i_e] = f
