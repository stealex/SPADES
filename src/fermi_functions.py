import numpy as np
from numba import njit
from . import ph
from scipy import special
from .wavefunctions import wavefunctions_handler, bound_handler, scattering_handler
from scipy.interpolate import Akima1DInterpolator, CubicSpline
from scipy import integrate
from typing import Callable
from functools import lru_cache

from mpmath import mp, hyp1f1, mpc


class fermi_functions:
    def __init__(self) -> None:
        pass

    def ff0_eval(self, ke: float):
        pass

    def ff1_eval(self, ke: float):
        pass


class point_like(fermi_functions):
    def __init__(self, z: int, r: float, e_grid: np.ndarray | None = None) -> None:
        self.z = z
        self.r = r

        if not (e_grid is None):
            gm1 = np.zeros_like(e_grid)
            fp1 = np.zeros_like(e_grid)

            for ie in range(len(e_grid)):
                gm1[ie] = self.gk(e_grid[ie], -1)
                fp1[ie] = self.gk(e_grid[ie], -1)

                self.ff0 = CubicSpline(
                    e_grid,
                    np.abs(gm1*gm1) + np.abs(fp1*fp1)
                )
                self.ff1 = CubicSpline(
                    e_grid,
                    2.0*np.real(gm1*np.conj(fp1))
                )

    def gk(self, ke, kappa):
        r = self.r
        gamma = np.sqrt(1.0-(ph.fine_structure*self.z)**2.0)
        p = np.sqrt(ke*(ke+2.*ph.electron_mass))
        e = ke+ph.electron_mass
        eta = ph.fine_structure*self.z*e/p

        g1 = np.abs(special.gamma(1.+gamma+1.0j*eta))
        g2 = special.gamma(1.+2.*gamma)
        eixsi = np.sqrt((kappa-1.0j*eta*ph.electron_mass/e)/(gamma-1.0j*eta))
        f11 = complex(mpc(hyp1f1(gamma-1.0j*eta, 1+2*gamma, -2.0j*p*r/ph.hc)))

        result = kappa/np.abs(kappa)*(1./(p*r/ph.hc)) *\
            np.sqrt((e+ph.electron_mass)/(2*e)) *\
            np.imag(np.exp(1.0j*p*r/ph.hc)*eixsi*f11) *\
            g1/g2 *\
            ((2*p*r/ph.hc)**(gamma)) *\
            np.exp(np.pi*eta/2.)
        return result

    def fk(self, ke, kappa):
        r = self.r
        gamma = np.sqrt(1.0-(ph.fine_structure*self.z)**2.0)
        p = np.sqrt(ke*(ke+2.*ph.electron_mass))
        e = ke+ph.electron_mass
        eta = ph.fine_structure*self.z*e/p

        g1 = np.abs(special.gamma(1.+gamma+1.0j*eta))
        g2 = special.gamma(1.+2.*gamma)
        eixsi = np.sqrt((kappa-1.0j*eta*ph.electron_mass/e)/(gamma-1.0j*eta))
        f11 = complex(mpc(hyp1f1(gamma-1.0j*eta, 1+2*gamma, -2.0j*p*r/ph.hc)))

        result = kappa/np.abs(kappa)*(1./(p*r/ph.hc)) *\
            np.sqrt((e-ph.electron_mass)/(2*e)) *\
            np.real(np.exp(1.0j*p*r/ph.hc)*eixsi*f11) *\
            g1/g2 *\
            ((2*p*r/ph.hc)**(gamma)) *\
            np.exp(np.pi*eta/2.)
        return result

    @ lru_cache(maxsize=None)
    def ff0_eval(self, ke: float):
        gm1 = self.gk(ke, -1)
        fp1 = self.fk(ke, 1)
        return np.abs(gm1*gm1) + np.abs(fp1*fp1)

    @ lru_cache(maxsize=None)
    def ff1_eval(self, ke: float):
        gm1 = self.gk(ke, -1)
        fp1 = self.fk(ke, 1)
        return 2.0*np.real(gm1*np.conj(fp1))


class charged_sphere(fermi_functions):
    def __init__(self, z: int, r: float) -> None:
        self.z = z
        self.r = r

    @ lru_cache(maxsize=None)
    def ff0_eval(self, ke):
        gamma = np.sqrt(1.0-(ph.fine_structure*self.z)**2.0)
        p = np.sqrt(ke*(ke+2*ph.electron_mass))
        eta = ph.fine_structure*self.z*(ke+ph.electron_mass)/p

        g1 = special.gamma(gamma+1.0j*eta)
        g2 = special.gamma(2*gamma+1)

        # g3 = (1.0+gamma)/2.
        g3 = 4.
        ff = g3*(2.0*p*self.r/ph.hc)**(2*(gamma-1)) * \
            np.exp(np.pi*eta)*np.abs((g1/g2)**2.0)

        # ff = ((g3/g2)**2.0) * (2.0*p*self.r/ph.hc)**(2.0 *
        #                                              (gamma-1))*(np.abs(g1)**2.0)*np.exp(np.pi*eta)
        return ff

    @ lru_cache(maxsize=None)
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
    def __init__(self, scattering_handler: scattering_handler, radius: float, density_function: Callable | None = None) -> None:
        self.scattering_handler = scattering_handler
        self.f = {}
        self.g = {}
        e = self.scattering_handler.energy_grid
        p = np.sqrt(e*(e+2.0*ph.electron_mass))
        norm = self.scattering_handler.norm
        pr = p*radius/ph.hc

        for k in self.scattering_handler.p_grid:
            self.f[k] = np.zeros(
                len(self.scattering_handler.energy_grid), dtype=complex)
            self.g[k] = np.zeros_like(self.f[k])
            phase_tot = self.scattering_handler.phase_grid[k] + \
                self.scattering_handler.coul_phase_grid[k]

            for i_e in range(len(e)):
                p_func = CubicSpline(
                    self.scattering_handler.r_grid, self.scattering_handler.p_grid[k][i_e]
                )
                q_func = CubicSpline(
                    self.scattering_handler.r_grid, self.scattering_handler.q_grid[k][i_e]
                )
                if density_function is None:
                    self.g[k][i_e] = norm[i_e]/(pr[i_e]) * \
                        p_func(radius) * np.exp(-1.0j*phase_tot[i_e])
                    self.f[k][i_e] = norm[i_e]/(pr[i_e]) * \
                        q_func(radius) * np.exp(-1.0j*phase_tot[i_e])

                else:
                    for i_e in range(len(e)):
                        p_func = Akima1DInterpolator(
                            self.scattering_handler.r_grid, self.scattering_handler.p_grid[k][i_e]
                        )
                        q_func = Akima1DInterpolator(
                            self.scattering_handler.r_grid, self.scattering_handler.p_grid[k][i_e]
                        )

                        p_avg = integrate.quad(lambda r: p_func(r)*density_function(r),
                                               self.scattering_handler.r_grid[0],
                                               self.scattering_handler.r_grid[-1])
                        q_avg = integrate.quad(lambda r: q_func(r)*density_function(r),
                                               self.scattering_handler.r_grid[0],
                                               self.scattering_handler.r_grid[-1])
                        self.f[k][i_e] = norm[i_e] / \
                            (pr[i_e]) * \
                            np.exp(-1.0j*phase_tot[i_e]) * q_avg
                        self.g[k][i_e] = norm[i_e] / \
                            (pr[i_e]) * \
                            np.exp(-1.0j*phase_tot[i_e]) * p_avg

        self.build_fermi_functions()

    @ lru_cache(maxsize=None)
    def ff0_eval(self, energy):
        return self.ff0(energy)

    @ lru_cache(maxsize=None)
    def ff1_eval(self, energy):
        return self.ff1(energy)

    def build_fermi_functions(self):
        self.gm1 = self.g[-1]
        self.fp1 = self.f[1]
        self.ff0 = CubicSpline(
            self.scattering_handler.energy_grid,
            np.abs(self.gm1*self.gm1) + np.abs(self.fp1*self.fp1)
        )
        self.ff1 = CubicSpline(
            self.scattering_handler.energy_grid,
            2.0*np.real(self.gm1*np.conj(self.fp1))
        )
