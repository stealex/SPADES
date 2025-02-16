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


def test_phase_shits_well():

    # v0 = -75.187/ph.hartree_energy  # MeV
    # r1 = 8.0*ph.fermi/ph.bohr_radius  # fm
    # r_inf = 30.0*ph.fermi/ph.bohr_radius  # fm
    # kappa = -1

    # fig 7
    v0 = -81.505/ph.hartree_energy  # MeV
    r1 = 15.0*ph.fermi/ph.bohr_radius  # fm
    r2 = 15.0*ph.fermi/ph.bohr_radius  # fm
    r_inf = 30.0*ph.fermi/ph.bohr_radius  # fm
    kappa = 1

    # v0 = 1.185  # MeV
    # r0 = 180.0  # fm
    # kappa = -1

    # mass_ratio = 1.  # 0.5/ph.electron_mass
    # v0 = (-76.205/mass_ratio)/ph.hartree_energy  # MeV
    # r0 = 8.0*mass_ratio*ph.fermi/ph.bohr_radius  # fm
    # kappa = 1

    # v0 = -108.205/ph.hartree_energy  # MeV
    # r1 = 8.0*ph.fermi/ph.bohr_radius  # fm
    # r2 = 15.0 * ph.fermi/ph.bohr_radius  # fm
    # r_inf = 30.0*ph.fermi/ph.bohr_radius  # fm
    # kappa = 1
    # v0 = v0/ph.hartree_energy  # MeV
    # r1 = r1*ph.fermi/ph.bohr_radius  # fm
    # r_inf = r_inf*ph.fermi/ph.bohr_radius  # fm

    # make spherical well potential
    r_points_1 = np.linspace(0., r1, 1000)
    v_points_1 = v0*np.ones_like(r_points_1)

    r_points_2 = np.linspace(r1, r_inf, 1000)
    v_points_2 = np.zeros_like(r_points_2)

    r_points = np.concatenate((r_points_1, r_points_2))
    v_points = np.concatenate((v_points_1, v_points_2))

    fig, ax = plt.subplots()
    ax.plot(r_points, v_points)
    # plt.show()

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

    energy_vals = np.linspace(0.01, 1.0, 200)
    delta_vals = np.zeros_like(energy_vals)
    for i_e in range(len(energy_vals)):
        e = (energy_vals[i_e]/ph.hartree_energy)
        delta_vals[i_e] = radial_wrapper.call_dfree(e, kappa, 1E-14)
        p, q = radial_wrapper.call_getpq(len(r))
        fig, ax = plt.subplots()
        print(e, e*ph.hartree_energy, delta_vals[i_e])

        i_last = radial_wrapper.call_getilast()
        ax.plot(r, p)
        ax.plot(r[:i_last], q[:i_last])
        ax.plot(r[i_last:], q[i_last:])  # *q[i_last-1]/q[i_last])
        # ax.axvline(r[i_last], color="red")
        # ax.set_xscale("log")
        plt.show()

    k_vals = np.sqrt((energy_vals+ph.electron_mass)**2.0-ph.electron_mass**2.0)
    fig, ax = plt.subplots(ncols=3, figsize=(16, 6))

    ax[0].plot(k_vals/ph.electron_mass, delta_vals)
    ax[1].plot(energy_vals, np.sin(delta_vals)**2.0)

    return (energy_vals, delta_vals)


def test_phase_shifts_barrier():
    pass


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


def num_int():
    e_kin = np.linspace(0.001, 1.0, 1000)
    e_vals = e_kin + ph.electron_mass

    # fig 6
    v0 = -75.187  # MeV
    r1 = 8.0  # fm
    r2 = 8.0
    r_inf = 30.0  # fm
    kappa = -1

    # v0 = 4.3*ph.electron_mass
    # rb = ph.hc/ph.electron_mass
    # r_inf = 1000.0
    # kappa = -1

    # mass_ratio = 1.  # 0.5/ph.electron_mass
    # v0 = (-76.205/mass_ratio)/ph.hartree_energy  # MeV
    # r0 = 8.0*mass_ratio*ph.fermi/ph.bohr_radius  # fm
    # kappa = 1

    # fig 7
    # v0 = -81.505  # MeV
    # r1 = 15.0  # fm
    # r2 = 15.0  # fm
    # r_inf = 30.0  # fm
    # kappa = 1

    delta_vals = np.zeros_like(e_vals)
    k_vals = np.sqrt(e_vals**2.0-ph.electron_mass**2.0)

    for i_e in range(len(e_vals)):
        e = e_vals[i_e]
        k = np.sqrt(e*e-ph.electron_mass**2.0)
        l_kappa = kappa if kappa > 0 else -kappa-1
        l_mkappa = -kappa if -kappa > 0 else kappa-1
        jl = spherical_jn(l_kappa, k*r_inf/ph.hc)
        jml = spherical_jn(l_mkappa, k*r_inf/ph.hc)
        yl = spherical_yn(l_kappa, k*r_inf/ph.hc)
        yml = spherical_yn(l_mkappa, k*r_inf/ph.hc)

        fact_g_extern = kappa/np.abs(kappa) * \
            (k*r_inf)/((e+ph.electron_mass)/1.)
        sol_a = solve_ivp(dirac_eq, (r_inf, r2), (r_inf*jl, fact_g_extern*jml),
                          args=(e, lambda x: 0., kappa), t_eval=np.linspace(r_inf, r2, 1000))
        sol_b = solve_ivp(dirac_eq, (r_inf, r2), (r_inf*yl, fact_g_extern*yml),
                          args=(e, lambda x: 0., kappa), t_eval=np.linspace(r_inf, r2, 1000))

        p = dirac_p(e, v0)
        r0 = 1E-3
        jl_zero = spherical_jn(l_kappa, p*r0/ph.hc)
        jml_zero = spherical_jn(l_mkappa, p*r0/ph.hc)
        fact_g_intern = kappa/np.abs(kappa) * \
            (p*r_inf/ph.hc)/((e-v0+ph.electron_mass))
        sol_int = solve_ivp(dirac_eq, (r0, r2), (r0*jl_zero, fact_g_intern*jml_zero),
                            args=(e, lambda x: v_func(x, v0), kappa),
                            t_eval=np.linspace(r0, r2, 1000))
        if (i_e == 0):
            r_grid_intern = sol_int.t
            r_grid_extern = sol_a.t
            v_grid_intern = v0*np.ones_like(r_grid_intern)
            v_grid_extern = np.zeros_like(r_grid_extern)

        coef_array = np.array(
            [[sol_a.y[0][-1], sol_b.y[0][-1]],
             [sol_a.y[1][-1], sol_b.y[1][-1]]])

        free_term = np.array([sol_int.y[0][-1], sol_int.y[1][-1]])

        # solve coeff_array * (b1, b2) = free_term
        b = np.linalg.solve(coef_array, free_term)
        # print(b)
        delta_vals[i_e] = np.arctan2(-b[1], b[0])

    fig, ax = plt.subplots(ncols=2, figsize=(16, 6))
    delta_vals_ana = dirac_phase_sphift(
        e_vals, r0, v0, kappa)
    ax[0].plot(k_vals/ph.electron_mass, delta_vals)
    # ax[0].plot(k_vals/ph.electron_mass, delta_vals_ana)
    ax[1].plot(e_vals, np.sin(delta_vals)**2.0)
    ax[1].set_xlim(0.800, 1.)

    fig, ax = plt.subplots()
    ax.scatter(r_grid_intern, v_grid_intern)
    ax.scatter(r_grid_extern, v_grid_extern)
    # legend and labels
    ax.legend(["Intern", "Extern"])
    ax.set_xlabel("r [fm]")
    ax.set_ylabel("V [MeV]")

    ind = np.argmin(np.abs(k_vals - 2*ph.electron_mass))
    print(k_vals[ind]/ph.electron_mass, delta_vals[ind])

    return (e_vals, delta_vals)


def make_demorad_input_schrod():
    k_vals = np.linspace(0.0001, 20.0, 100)
    v0 = 2.4  # MeV
    r0 = 1.*ph.hc
    l = 0
    mass_ratio = 0.5/ph.electron_mass
    print(f"mass_ratio = {mass_ratio}")
    with open("demorad_input.dat", "w") as f:
        f.write("1\n")
        f.write(
            f"0 {-v0/mass_ratio/ph.hartree_energy} {r0*mass_ratio*ph.fermi/ph.bohr_radius}\n")
        f.write("n\n")
        for i_e in range(len(k_vals)):
            k = k_vals[i_e]
            e = k*k/(2*0.5)
            f.write("2\n")
            f.write(f"{e/mass_ratio/ph.hartree_energy} {l} {1E-14}\n")
        f.write("-1\n")


def make_demorad_input():
    k_vals = np.linspace(0.0001, 20.0, 10000)
    m = 0.5
    v0 = 4.195*m  # MeV
    r0 = 1./m*ph.hc
    kappa = -1
    mass_ratio = m/ph.electron_mass
    print(f"mass_ratio = {mass_ratio}")
    with open("demorad_input.dat", "w") as f:
        f.write("1\n")
        f.write(
            f"0 {-v0/mass_ratio/ph.hartree_energy} {r0*mass_ratio*ph.fermi/ph.bohr_radius}\n")
        f.write("n\n")
        for i_e in range(len(k_vals)):
            k = k_vals[i_e]
            e = k*k/(2*m)
            f.write("4\n")
            f.write(f"{e/mass_ratio/ph.hartree_energy} {kappa} {1E-14}\n")
        f.write("-1\n")


def num_int_schrod():
    e_kin = np.linspace(0.0001, 40.0, 1000)
    e_vals = e_kin

    # fig 6
    v0 = -2.4*ph.electron_mass/0.5  # MeV
    rb = 1.0*ph.hc*0.5/ph.electron_mass  # fm
    r_inf = 1000.0  # fm
    l = 0

    # v0 = 4.3*ph.electron_mass
    # rb = ph.hc/ph.electron_mass
    # r_inf = 1000.0
    # kappa = -1
    # fig 7
    # v0 = 76.205  # MeV
    # rb = 8.0  # fm
    # r_inf = 30.0  # fm
    # kappa = 1

    delta_vals = np.zeros_like(e_vals)
    k_vals = np.sqrt(2.0*ph.electron_mass*e_vals)
    for i_e in range(len(e_vals)):
        e = e_vals[i_e]
        k = k_vals[i_e]
        jl = spherical_jn(l, k*r_inf/ph.hc)
        jl_prime = k/ph.hc*spherical_jn(l, k*r_inf/ph.hc, derivative=True)
        yl = spherical_yn(l, k*r_inf/ph.hc)
        yl_prime = k/ph.hc*spherical_yn(l, k*r_inf/ph.hc, derivative=True)

        sol_a = solve_ivp(schrod_eq, (r_inf, rb), (jl, jl_prime),
                          args=(e, lambda x: 0., l), t_eval=np.linspace(r_inf, rb, 1000))
        sol_b = solve_ivp(schrod_eq, (r_inf, rb), (yl, yl_prime),
                          args=(e, lambda x: 0., l), t_eval=np.linspace(r_inf, rb, 1000))

        p = np.sqrt(2.0*ph.electron_mass*(e-v0))
        r0 = 1E-3
        jl_zero = spherical_jn(l, p*r0/ph.hc)
        jl_zero_prime = p/ph.hc*spherical_jn(l, p*r0/ph.hc, derivative=True)
        sol_int = solve_ivp(schrod_eq, (r0, rb), (jl_zero, jl_zero_prime),
                            args=(e, lambda x: v_func(x, v0), l),
                            t_eval=np.linspace(r0, rb, 1000))

        coef_array = np.array(
            [[sol_a.y[0][-1], sol_b.y[0][-1]],
             [sol_a.y[1][-1], sol_b.y[1][-1]]])

        free_term = np.array([sol_int.y[0][-1], sol_int.y[1][-1]])

        # solve coeff_array * (b1, b2) = free_term
        b = np.linalg.solve(coef_array, free_term)
        # print(b)
        delta_vals[i_e] = np.arctan2(-b[1], b[0])

        if (i_e == 100):
            print(f"energy = {e}")
            fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(16, 6))
            jl_vals = spherical_jn(l, k*sol_a.t/ph.hc)
            ax[0, 0].plot(sol_a.t, sol_a.y[0])
            ax[0, 0].scatter(sol_a.t[0:-1:20], jl_vals[0:-1:20])

            jl_prime_vals = spherical_jn(l, k*sol_a.t/ph.hc, derivative=True)
            ax[0, 1].plot(sol_a.t, sol_a.y[1])
            ax[0, 1].scatter(sol_a.t[0:-1:20], k/ph.hc*jl_prime_vals[0:-1:20])

            yl_vals = spherical_yn(l, k*sol_a.t/ph.hc)
            ax[1, 0].plot(sol_b.t, sol_b.y[0])
            ax[1, 0].scatter(sol_b.t[0:-1:20], yl_vals[0:-1:20])

            yl_prime_vals = spherical_yn(l, k*sol_a.t/ph.hc, derivative=True)
            ax[1, 1].plot(sol_b.t, sol_b.y[1])
            ax[1, 1].scatter(sol_b.t[0:-1:20], k/ph.hc*yl_prime_vals[0:-1:20])

            fig, ax = plt.subplots(ncols=2, figsize=(16, 6))
            jl_vals = spherical_jn(l, p*sol_int.t/ph.hc)
            ax[0].plot(sol_int.t, sol_int.y[0])
            ax[0].scatter(sol_int.t[0:-1:20], jl_vals[0:-1:20])

            jl_prime_vals = spherical_jn(l, p*sol_int.t/ph.hc, derivative=True)
            ax[1].plot(sol_int.t, sol_int.y[1])
            ax[1].scatter(sol_int.t[0:-1:20], p /
                          ph.hc*jl_prime_vals[0:-1:20])

    fig, ax = plt.subplots(ncols=2, figsize=(16, 6))
    ax[0].plot(k_vals*0.5/ph.electron_mass, delta_vals)
    ax[1].plot(e_vals, np.sin(delta_vals)**2.0)
    # ax[1].set_xlim(0.800, 1.)


if __name__ == "__main__":
    e_rad, delta_rad = test_phase_shits_well()
    plt.show()
    e_num, delta_num = num_int()

    # plot the two
    fig, ax = plt.subplots(ncols=2, figsize=(16, 6))
    ax[0].plot(e_rad, delta_rad)
    ax[0].plot(e_num-ph.electron_mass, delta_num)
    ax[0].legend(["Radial", "RK"])
    # labels
    ax[0].set_xlabel("Energy [MeV]")
    ax[0].set_ylabel("Phase shift [rad]")

    ax[1].plot(e_rad, np.sin(delta_rad)**2.0)
    ax[1].plot(e_num-ph.electron_mass, np.sin(delta_num)**2.0)
    ax[1].legend(["Radial", "RK"])
    # labels
    ax[1].set_xlabel("Energy [MeV]")
    ax[1].set_ylabel("sin(delta)")
    # num_int_schrod()
    # make_demorad_input()
    plt.show()
