from matplotlib import axes
from matplotlib.pylab import f
import numpy as np
from src import radial_wrapper, ph
from src.wavefunctions import scattering_config, scattering_handler
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn
from scipy.integrate import solve_ivp


def dirac_p(e, v):
    return np.sqrt((e-v)**2.0 - ph.electron_mass**2.0)


def dgamma(e, v):
    p = dirac_p(e, v)
    k = np.sqrt(e*e-ph.electron_mass**2.0)
    return p/k * (e+ph.electron_mass)/(e-v+ph.electron_mass)


def dirac_phase_sphift(e, a, v0, kappa):
    gam = dgamma(e, v0)
    k = np.sqrt(e*e-ph.electron_mass**2.0)
    p = dirac_p(e, v0)
    l_kappa = kappa if kappa > 0 else -kappa-1
    l_mkappa = -kappa if -kappa > 0 else kappa-1
    num = gam*spherical_jn(l_kappa, k*a/ph.hc)*spherical_jn(l_mkappa, p*a/ph.hc) - \
        spherical_jn(l_kappa, p*a/ph.hc)*spherical_jn(l_mkappa, k*a/ph.hc)
    den = gam*spherical_jn(l_mkappa, p*a/ph.hc)*spherical_yn(l_kappa, k*a/ph.hc) -\
        spherical_jn(l_kappa, p*a/ph.hc)*spherical_yn(l_mkappa, k*a/ph.hc)

    return np.arctan2(num, den)


def test_phase_shits_well(e, kappa, vv0, rr1, rr_inf):

    v0 = vv0/ph.hartree_energy  # MeV
    r1 = rr1*ph.fermi/ph.bohr_radius  # fm
    r_inf = rr_inf*ph.fermi/ph.bohr_radius  # fm
    # make spherical well potential
    r_points_1 = np.linspace(0., r1, 1000)
    v_points_1 = v0*np.ones_like(r_points_1)

    r_points_2 = np.linspace(r1, r_inf, 1000)
    v_points_2 = np.zeros_like(r_points_2)

    r_points = np.concatenate((r_points_1, r_points_2))
    v_points = np.concatenate((v_points_1, v_points_2))

    radial_wrapper.call_vint(r_points, r_points*v_points)
    # _, r, dr = radial_wrapper.call_sgrid(r_points[-1],
    #                                      r_points[1],
    #                                      r_points[-1]-r_points[-3],
    #                                      5000,
    #                                      20000)
    # r = np.append(r, r1*ph.fermi/ph.bohr_radius)
    # r.sort()
    r = np.concatenate((r_points_1, r_points_2[1:]))
    radial_wrapper.call_setrgrid(r)

    e_hartree = e/ph.hartree_energy
    delta = radial_wrapper.call_dfree(e_hartree, kappa, 1E-14)
    p, q = radial_wrapper.call_getpq(len(r))
    i_last = radial_wrapper.call_getilast()

    return (delta, p, q, i_last, r*ph.bohr_radius/ph.fermi)


def dirac_eq(x, y, energy, v, kappa):
    f = y[0]
    g = y[1]

    fprime = -kappa*f/x + (energy+ph.electron_mass-v(x))*g/ph.hc
    gprime = kappa*g/x - (energy-ph.electron_mass-v(x))*f/ph.hc

    return (fprime, gprime)


def schrod_eq(x, y, energy, v, l):
    f = y[0]
    g = y[1]

    fprime = g
    gprime = 2*ph.electron_mass*energy / \
        (ph.hc**2.0)*(v(x)/energy - 1.)*f + l*(l+1)/(x*x) * f -\
        2/x*fprime

    return (fprime, gprime)


def v_func(x, v0):
    return v0


def num_int(ek_val, kappa, vv0, rr1, rr_inf):
    e = ek_val + ph.electron_mass

    # fig 6
    v0 = vv0  # MeV
    r1 = rr1  # fm
    r_inf = rr_inf  # fm

    k = np.sqrt(e*e-ph.electron_mass**2.0)
    l_kappa = kappa if kappa > 0 else -kappa-1
    l_mkappa = -kappa if -kappa > 0 else kappa-1
    jl = spherical_jn(l_kappa, k*r_inf/ph.hc)
    jml = spherical_jn(l_mkappa, k*r_inf/ph.hc)
    yl = spherical_yn(l_kappa, k*r_inf/ph.hc)
    yml = spherical_yn(l_mkappa, k*r_inf/ph.hc)

    fact_g_extern = kappa/np.abs(kappa) * \
        (k*r_inf)/((e+ph.electron_mass)/1.)
    sol_a = solve_ivp(dirac_eq, (r_inf, r1), (r_inf*jl, fact_g_extern*jml),
                      args=(e, lambda x: 0., kappa), t_eval=np.linspace(r_inf, r1, 1000))
    sol_b = solve_ivp(dirac_eq, (r_inf, r1), (r_inf*yl, fact_g_extern*yml),
                      args=(e, lambda x: 0., kappa), t_eval=np.linspace(r_inf, r1, 1000))

    p = dirac_p(e, v0)
    r0 = 1E-3
    jl_zero = spherical_jn(l_kappa, p*r0/ph.hc)
    jml_zero = spherical_jn(l_mkappa, p*r0/ph.hc)
    fact_g_intern = kappa/np.abs(kappa) * \
        (p*r0/ph.hc)/((e-v0+ph.electron_mass))
    sol_int = solve_ivp(dirac_eq, (r0, r1), (r0*jl_zero, fact_g_intern*jml_zero),
                        args=(e, lambda x: v_func(x, v0), kappa),
                        t_eval=np.linspace(r0, r1, 1000))

    r_grid_intern = sol_int.t
    r_grid_extern = sol_a.t
    r_grid_tot = np.concatenate((r_grid_intern, r_grid_extern[-1::-1][1:]))

    coef_array = np.array(
        [[sol_a.y[0][-1], sol_b.y[0][-1]],
         [sol_a.y[1][-1], sol_b.y[1][-1]]])

    free_term = np.array([sol_int.y[0][-1], sol_int.y[1][-1]])

    # solve coeff_array * (b1, b2) = free_term
    b = np.linalg.solve(coef_array, free_term)
    delta = np.arctan2(-b[1], b[0])
    tt = np.abs(delta)
    if (np.abs(delta) > np.pi/2):
        delta = delta*(1.0-np.pi/tt)

    cd = np.cos(delta)
    sd = np.sin(delta)
    rnorm = (cd * sol_a.y[0][-1] + sd * sol_b.y[0][-1])/sol_int.y[0][-1]
    # print(f"rnorm = {rnorm}")
    p_int = sol_int.y[0]
    q_int = sol_int.y[1]

    p_extern = (b[0] * sol_a.y[0] + b[1] * sol_b.y[0])
    q_extern = (b[0] * sol_a.y[1] + b[1] * sol_b.y[1])

    p = np.concatenate((p_int, p_extern[-1::-1][1:]))
    q = np.concatenate((q_int, q_extern[-1::-1][1:]))

    return (delta, p, q, r_grid_tot)


if __name__ == "__main__":
    v0 = -81.505  # MeV
    r1 = 15.0  # fm
    r2 = 15.0  # fm
    r_inf = 30.0  # fm
    kappa = 1

    # v0 = -76.205  # MeV
    # r1 = 8.0  # fm
    # r2 = 8.0  # fm
    # r_inf = 30.0  # fm
    # kappa = 1

    # v0 = -75.187  # MeV
    # r1 = 8.0  # fm
    # r2 = 8.0  # fm
    # r_inf = 30.0  # fm
    # kappa = -1

    e_grid = np.linspace(0.01, 1.0, 1000)
    delta_rad_grid = np.zeros_like(e_grid)
    delta_rk_grid = np.zeros_like(e_grid)
    delta_ana = np.zeros_like(e_grid)
    for i_e in range(len(e_grid)):
        e_trial = e_grid[i_e]
        delta_rad, p_rad, q_rad, i_last, r_rad = test_phase_shits_well(
            e_trial, kappa, v0, r1, r_inf)
        delta_rk, p_rk, q_rk, r_rk = num_int(e_trial, kappa, v0, r1, r_inf)

        delta_rad_grid[i_e] = delta_rad
        delta_rk_grid[i_e] = delta_rk
        d_ana = dirac_phase_sphift(
            e_trial+ph.electron_mass, r1, v0, kappa)
        tt = np.abs(d_ana)
        if (np.abs(d_ana) > np.pi/2):
            d_ana = d_ana*(1.0-np.pi/tt)
        delta_ana[i_e] = d_ana
        # fig, ax = plt.subplots(ncols=2)
        # ax[0].plot(r_rad, p_rad, label='p_rad')
        # ax[0].plot(r_rad, q_rad, label='q_rad')

        # ax[1].plot(r_rk, p_rk, label='p_rk')
        # ax[1].plot(r_rk, q_rk, label='q_rk')

        # print(delta_rad, delta_rk)
        # plt.show()

    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(e_grid+ph.electron_mass, delta_rad_grid, label="RADIAL")
    ax[0].plot(e_grid+ph.electron_mass, delta_rk_grid,
               label="RK", linestyle="--")
    # dashed line
    ax[0].plot(e_grid+ph.electron_mass, delta_ana,
               label="ANALYTIC", linestyle=':')
    ax[0].set_xlabel("E [MeV]")
    ax[0].set_ylabel(r"$\delta$ [rad]")
    ax[0].legend()

    ax[1].plot(e_grid+ph.electron_mass,
               np.sin(-delta_rad_grid)**2.0, label="RADIAL")
    ax[1].plot(e_grid+ph.electron_mass, np.sin(delta_rk_grid)
               ** 2.0, label="RK", linestyle="--")
    ax[1].plot(e_grid+ph.electron_mass, np.sin(delta_ana)
               ** 2.0, label="ANALYTIC", linestyle=':')
    ax[1].set_xlabel("E [MeV]")
    ax[1].set_ylabel(r"$\sin(\delta)$")
    ax[1].set_xlim(0.8, 1.0)
    ax[1].legend()
    plt.show()
