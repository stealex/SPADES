"""Exchange-correction utilities built from bound/scattering wavefunction overlaps."""

from .wavefunctions import WaveFunctionsHandler
import numpy as np
from scipy.interpolate import CubicSpline
from scipy import integrate
from . import ph
import logging

logger = logging.getLogger(__name__)


def integrate_wf(f1: np.ndarray,
                 f2: np.ndarray,
                 r1: np.ndarray,
                 r2: np.ndarray):
    """Integrate product of two radial wavefunction components on a common grid.

    Parameters
    ----------
    f1, f2:
        Radial arrays to multiply point-wise.
    r1, r2:
        Radial grids for ``f1`` and ``f2``. Currently the integral is evaluated on ``r2``.

    Returns
    -------
    float
        Simpson integral of the product.
    """
    # test with only interpolation
    # scattering_func = Akima1DInterpolator(r1, f1)
    # bound_func = Akima1DInterpolator(r2, f2)
    # integrant = np.array([scattering_func(x)*bound_func(x)
    #                      for x in r2])

    integrant = f1*f2
    result = integrate.simpson(
        y=integrant,
        x=r2
    )

    return result


class ExchangeCorrection:
    """Compute exchange-correction factors for beta-decay observables."""

    def __init__(self, initial_handler: WaveFunctionsHandler, final_handler: WaveFunctionsHandler, nuclear_radius: float) -> None:
        """Store initial/final wavefunction handlers and nuclear-size scale.

        Parameters
        ----------
        initial_handler:
            Wavefunction handler for the initial atom (bound states).
        final_handler:
            Wavefunction handler for the final atom (bound + scattering states).
        nuclear_radius:
            Nuclear radius in fm.
        """
        self.initial_handler = initial_handler
        self.final_handler = final_handler
        self.nuclear_radius = nuclear_radius

    def compute_overlaps(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Compute scattering-bound and bound-bound overlaps for one ``kappa``.

        Parameters
        ----------
        k:
            Relativistic angular quantum number.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(scattering_bound, bound_bound)`` overlap arrays.
        """
        logger.debug("Computing overlaps for k=%d", k)
        n_vals = self.initial_handler.bound_config.n_values
        energy_vals = self.final_handler.scattering_handler.energy_grid

        scattering_bound = np.zeros((len(energy_vals), len(n_vals)))
        bound_bound = np.zeros((len(n_vals)))

        for i_n in range(len(n_vals)):
            n = self.initial_handler.bound_config.n_values[i_n]
            if k not in self.initial_handler.bound_config.k_values[n]:
                continue

            for i_e in range(len(self.final_handler.scattering_handler.energy_grid)):

                gg = integrate_wf(self.final_handler.scattering_handler.p_grid[k][i_e],
                                  self.initial_handler.bound_handler.p_grid[n][k],
                                  self.final_handler.scattering_handler.r_grid,
                                  self.initial_handler.bound_handler.r_grid)
                ff = integrate_wf(self.final_handler.scattering_handler.q_grid[k][i_e],
                                  self.initial_handler.bound_handler.q_grid[n][k],
                                  self.final_handler.scattering_handler.r_grid,
                                  self.initial_handler.bound_handler.r_grid)

                scattering_bound[i_e, i_n] = gg+ff

            gg = integrate_wf(self.final_handler.bound_handler.p_grid[n][k],
                              self.initial_handler.bound_handler.p_grid[n][k],
                              self.final_handler.bound_handler.r_grid,
                              self.initial_handler.bound_handler.r_grid)
            ff = integrate_wf(self.final_handler.bound_handler.q_grid[n][k],
                              self.initial_handler.bound_handler.q_grid[n][k],
                              self.final_handler.bound_handler.r_grid,
                              self.initial_handler.bound_handler.r_grid)
            bound_bound[i_n] = gg+ff

        return (scattering_bound, bound_bound)

    def transform_scattering_wavefunctions(self):
        """Transformation according to AIP Conf. Proc. 3138, 020012 (2024)
        https://doi.org/10.1063/5.0205270

        
        Returns
        -------
        tuple[dict, dict]
            Transformed ``(p_grid, q_grid)`` dictionaries keyed by ``kappa``.
        """
        logger.info("Transforming scattering wavefunctions")
        n_values = self.initial_handler.bound_config.n_values
        k_values = self.initial_handler.bound_config.k_values

        p_new = {}
        q_new = {}
        for k in [-1, 1]:
            scattering_bound, bound_bound = self.compute_overlaps(k)
            p_new[k] = np.zeros_like(
                self.final_handler.scattering_handler.p_grid[k])
            q_new[k] = np.zeros_like(
                self.final_handler.scattering_handler.q_grid[k])
            for i_e in range(len(self.final_handler.scattering_handler.energy_grid)):
                p_new[k][i_e] = self.final_handler.scattering_handler.p_grid[k][i_e]
                q_new[k][i_e] = self.final_handler.scattering_handler.q_grid[k][i_e]
                for i_n in range(len(n_values)):
                    n = n_values[i_n]
                    if not (k in k_values[n]):
                        continue

                    p_new[k][i_e] = p_new[k][i_e] - scattering_bound[i_e][i_n]/bound_bound[i_n] * \
                        self.initial_handler.bound_handler.p_grid[n][k]
                    q_new[k][i_e] = q_new[k][i_e] - scattering_bound[i_e][i_n]/bound_bound[i_n] * \
                        self.initial_handler.bound_handler.q_grid[n][k]
        return p_new, q_new

    def compute_t(self):
        """Compute shell-dependent exchange amplitudes ``t_n``.

        Returns
        -------
        dict
            Nested dictionary ``t[n][kappa](E)``.
        """
        logger.info("Computing T_ns exchange amplitudes")
        n_values = self.initial_handler.bound_config.n_values
        k_values = self.initial_handler.bound_config.k_values

        self.t = {}
        r_nuc = 1.2 * \
            ((self.initial_handler.atomic_system.mass_number)**(1./3.))

        ir_nuc = np.abs(
            r_nuc - self.final_handler.bound_handler.r_grid).argmin()
        for k in [-1, 1]:
            scattering_bound, bound_bound = self.compute_overlaps(k)

            for i_n in range(len(n_values)):
                n = n_values[i_n]
                if not (k in k_values[n]):
                    continue

                if not (n in self.t):
                    self.t[n] = {}

                i_k = np.abs(
                    np.array(self.final_handler.scattering_config.k_values) - k).argmin()
                if k == -1:
                    g_prime_n = self.final_handler.bound_handler.p_grid[n][k][ir_nuc]
                    g_prime_e = self.final_handler.scattering_handler.p_grid[k][:, ir_nuc]
                    ratio = g_prime_n/g_prime_e
                else:
                    f_prime_n = self.final_handler.bound_handler.q_grid[n][k][ir_nuc]
                    f_prime_e = self.final_handler.scattering_handler.q_grid[k][:, ir_nuc]
                    ratio = f_prime_n/f_prime_e

                self.t[n][k] = -1.0*scattering_bound[:, i_n] / \
                    bound_bound[i_n] * ratio

        return self.t

    def compute_eta(self):
        """Compute partial exchange corrections for ``s`` and ``p`` channels.

        Returns
        -------
        dict
            Nested dictionary ``eta[n][kappa](E)``.
        """
        logger.info("Computing partial eta correction")
        try:
            tn = self.t
        except AttributeError:
            tn = self.compute_t()

        n_values = self.initial_handler.bound_config.n_values
        k_values = self.initial_handler.bound_config.k_values
        e_values = ph.electron_mass + \
            self.final_handler.scattering_handler.energy_grid

        r_nuc = 1.2 * \
            ((self.initial_handler.atomic_system.mass_number)**(1./3.))

        ir_nuc = np.abs(r_nuc -
                        self.final_handler.bound_handler.r_grid).argmin()

        gm1 = self.final_handler.scattering_handler.norm * \
            self.final_handler.scattering_handler.p_grid[-1][:, ir_nuc]
        fp1 = self.final_handler.scattering_handler.norm * \
            self.final_handler.scattering_handler.q_grid[1][:, ir_nuc]
        fs = gm1*gm1/(gm1*gm1+fp1*fp1)

        self.eta = {}
        for k in [-1, 1]:
            for n in n_values:
                if not (k in k_values[n]):
                    continue
                if not (n in self.eta):
                    self.eta[n] = {}

                if k == -1:
                    self.eta[n][k] = fs*(2*tn[n][k]+tn[n][k]*tn[n][k])
                else:
                    self.eta[n][k] = (1.-fs)*(2*tn[n][k]+tn[n][k]*tn[n][k])

        return self.eta

    def compute_eta_total(self):
        """Compute total exchange correction and interpolation function.

        Returns
        -------
        np.ndarray
            Total exchange correction sampled on the scattering-energy grid.
        """
        logger.info("Computing total eta correction")
        try:
            eta = self.eta
        except AttributeError:
            eta = self.compute_eta()

        self.eta_total = np.zeros_like(eta[1][-1])
        self.eta_s = np.zeros_like(self.eta_total)
        self.eta_p = np.zeros_like(self.eta_total)
        n_values = self.initial_handler.bound_config.n_values
        k_values = self.initial_handler.bound_config.k_values
        e_values = ph.electron_mass + \
            self.final_handler.scattering_handler.energy_grid

        r_nuc = 1.2 * \
            ((self.initial_handler.atomic_system.mass_number)**(1./3.))

        ir_nuc = np.abs(r_nuc -
                        self.final_handler.bound_handler.r_grid).argmin()

        gm1 = self.final_handler.scattering_handler.norm * \
            self.final_handler.scattering_handler.p_grid[-1][:, ir_nuc]
        fp1 = self.final_handler.scattering_handler.norm * \
            self.final_handler.scattering_handler.q_grid[1][:, ir_nuc]
        fs = gm1*gm1/(gm1*gm1+fp1*fp1)
        for n in n_values:
            self.eta_s = self.eta_s+eta[n][-1]
            if (1 in k_values[n]):
                self.eta_p = self.eta_p + eta[n][1]

            for m in n_values:
                if m == n:
                    continue
                self.eta_s = self.eta_s + self.t[n][-1]*self.t[m][-1]
                if 1 in k_values[m] and 1 in k_values[n]:
                    self.eta_p = self.eta_p + (1-fs)*self.t[n][1]*self.t[m][1]

        self.eta_total = self.eta_s+self.eta_p
        self.eta_total_func = CubicSpline(e_values-ph.electron_mass,
                                          self.eta_total)
        return self.eta_total
