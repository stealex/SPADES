import numpy as np
from numba import njit
from utils import ph
from scipy import special
from handlers import wavefunction_handlers
from scipy.interpolate import Akima1DInterpolator, CubicSpline
from scipy import integrate
from typing import Callable
from functools import lru_cache


class fermi_functions:
    def __init__(self) -> None:
        pass

    def ff0_eval(self, ke: float):
        pass

    def ff1_eval(self, ke: float):
        pass


class point_like(fermi_functions):
    def __init__(self, z: int, r: float) -> None:
        self.z = z
        self.r = r

    def ff0_eval(self, ke):
        gamma = np.sqrt(1.0-(ph.fine_structure*self.z)**2.0)
        p = np.sqrt(ke*(ke+2*ph.electron_mass))
        eta = ph.fine_structure*self.z*(ke+ph.electron_mass)/p

        g1 = special.gamma(gamma+1.0j*eta)
        g2 = special.gamma(2*gamma+1)

        g3 = 2.
        # ff = 4*(2.0*p*self.r/ph.hc)**(2*(gamma-1)) * \
        #     np.exp(np.pi*eta)*np.abs((g1/g2)**2.0)

        ff = ((g3/g2)**2.0) * (2.0*p*self.r/ph.hc)**(2.0 *
                                                     (gamma-1))*(np.abs(g1)**2.0)*np.exp(np.pi*eta)
        return ff

    def ff1_eval(self, ke):
        ff0 = self.ff0_eval(ke)
        # fact1 = np.sqrt((ke+2.0*ph.electron_mass)/(2.0*(ke+ph.electron_mass)))
        # gm1 = np.sqrt(ff0)*fact1
        # fact2 = np.sqrt(ke/(2.0*(ke+ph.electron_mass)))
        # fp1 = 1.0*np.sqrt(ff0)*fact2
        # return -2.0*np.real(gm1*np.conj(fp1))
        momentum = np.sqrt(ke*(ke+2.0*ph.electron_mass))
        e_total = (ke+ph.electron_mass)
        return momentum/e_total * ff0


class numeric:
    def __init__(self, scattering_handler: wavefunction_handlers.scattering_handler) -> None:
        self.scattering_handler = scattering_handler

    def eval_fg(self, radius: float, density_function: Callable | None = None):
        self.f = np.zeros((len(self.scattering_handler.scattering_config.k_values),
                           len(self.scattering_handler.energy_grid)),
                          dtype=complex)
        self.g = np.zeros_like(self.f)

        for i_k in range(len(self.scattering_handler.scattering_config.k_values)):
            if (density_function is None):
                id_rnuc = np.abs(
                    self.scattering_handler.r_grid*ph.bohr_radius/ph.fermi - radius).argmin()
                self.f[i_k] = self.scattering_handler.f_grid[i_k, :, id_rnuc]
                self.g[i_k] = self.scattering_handler.g_grid[i_k, :, id_rnuc]
            else:
                for i_e in range(len(self.scattering_handler.energy_grid)):
                    f_func = Akima1DInterpolator(
                        self.scattering_handler.r_grid, self.scattering_handler.f_grid[i_k, i_e])
                    g_func = Akima1DInterpolator(
                        self.scattering_handler.r_grid, self.scattering_handler.g_grid[i_k, i_e])

                    f_avg = integrate.quad(
                        lambda r: r*f_func(r)*density_function(r),
                        self.scattering_handler.r_grid[0],
                        self.scattering_handler.r_grid[-1]
                    )

                    g_avg = integrate.quad(
                        lambda r: r*g_func(r)*density_function(r),
                        self.scattering_handler.r_grid[0],
                        self.scattering_handler.r_grid[-1]
                    )

                    self.f[i_k, i_e] = f_avg
                    self.g[i_k, i_e] = g_avg

    @lru_cache(maxsize=None)
    def ff0_eval(self, energy):
        return self.ff0(energy/ph.hartree_energy)

    @lru_cache(maxsize=None)
    def ff1_eval(self, energy):
        return self.ff1(energy/ph.hartree_energy)

    def build_fermi_functions(self):
        k_vals = np.array(self.scattering_handler.scattering_config.k_values)
        km1 = np.abs(k_vals - (-1)).argmin()
        kp1 = np.abs(k_vals - 1).argmin()

        self.gm1 = self.g[km1]
        self.fp1 = self.f[kp1]
        self.ff0 = CubicSpline(
            self.scattering_handler.energy_grid,
            np.abs(self.gm1*self.gm1) + np.abs(self.fp1*self.fp1)
        )
        self.ff1 = CubicSpline(
            self.scattering_handler.energy_grid,
            2.0*np.real(self.gm1*np.conj(self.fp1))
        )
