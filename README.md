# SPADES

*Phase Space Factors and Spectra for Double Beta Decay*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12607448.svg)](https://doi.org/10.5281/zenodo.12607448)
[![Paper](https://img.shields.io/badge/Eur.Phys.J.C-86%3A496%20(2026)-blue)](https://doi.org/10.1140/epjc/s10052-026-15656-6)

SPADES computes phase-space factors (PSFs), single-electron and summed-electron
energy spectra, and angular correlation coefficients for double-beta decay (DBD),
starting from numerically exact Dirac wave functions in a self-consistent atomic
potential.

The lepton kinematics of DBD is not a solved-once problem: measured spectral
shapes are now precise enough that the atomic-physics ingredients — screening,
finite nuclear size, exchange with bound electrons, radiative corrections — and
the treatment of the virtual intermediate states shift predictions at a level
that experiments can see. SPADES makes those ingredients explicit and
individually switchable, so their impact can be quantified rather than assumed.

---

## Contents

- [What it computes](#what-it-computes)
- [Physics implementation](#physics-implementation)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration reference](#configuration-reference)
- [Output format](#output-format)
- [Implementation notes](#implementation-notes)
- [Batch and cluster runs](#batch-and-cluster-runs)
- [Repository layout](#repository-layout)
- [Units and conventions](#units-and-conventions)
- [Status and known limitations](#status-and-known-limitations)
- [Citing](#citing)
- [References](#references)
- [License](#license)

---

## What it computes

| Process | Config name | 2ν methods | 0ν mechanism |
|---|---|---|---|
| β⁻β⁻ | `2nu_2betaMinus` / `0nu_2betaMinus` | Closure, Taylor, CIFRA | light-neutrino exchange (`LNE`) |
| β⁺β⁺ | `2nu_2betaPlus` / `0nu_2betaPlus` | Closure, Taylor, CIFRA | `LNE` |
| ECβ⁺ | `2nu_ECbetaPlus` / `0nu_ECbetaPlus` | Closure | `LNE` |
| ECEC | `2nu_2EC` | Closure, Taylor | — |

Transitions: `0->0` (gs → gs), `0->2` (gs → first 2⁺), `0->02` (gs → excited 0⁺).

Observables per run:

- **`Single`** — single-lepton energy spectrum `dG/dε₁`
- **`Sum`** — summed-energy spectrum `dG/dT`
- **`Angular`** — angular-correlation spectrum `dH/dε₁`, from which
  `α(ε₁) = (H/G)·(dH/dε₁)/(dG/dε₁)`
- **PSFs** `G`, `H` obtained by integrating the above (reported in 1/year)
- Optional **2D spectra** `d²G/dε₁dε₂` on a hybrid log-linear grid, the input
  needed for event generators and detector Monte Carlo

## Physics implementation

**Atomic structure.** The initial atom is solved self-consistently with the
Dirac-Hartree-Fock-Slater (DHFS) method; the resulting potential is then used to
integrate the radial Dirac equation for both bound and continuum states with the
RADIAL package. Continuum states are computed for the *final* (daughter) ion, so
electron and positron wave functions are orthogonal to the bound spectrum of the
system they are actually emitted into — the point of departure from the usual
approximate treatments.

**Fermi functions.** Three levels of description are selectable and directly
comparable within one run:

- `PointLike` — analytic, point nucleus
- `ChargedSphere` — analytic, uniformly charged sphere
- `Numeric` — built from the numerically integrated continuum wave functions,
  evaluated at the nuclear surface (`OnSurface`), including screening, finite
  nuclear size and phase-shift effects

**Corrections.** `Exchange` (implemented as a transformation of the scattering
wave functions, following Nitescu et al.) and `Radiative` can be enabled
independently, which is what makes their individual contributions measurable.

**Intermediate states.** For 2ν modes, three treatments of the virtual
intermediate nucleus are available:

- **Closure** — a single effective energy `⟨E_N⟩ − E_I`, either given explicitly
  or from the systematics `Ã = 1.12·A^(1/2)`
- **Taylor** — expansion of the energy denominators to orders `0, 2, 22, 4`
  (`22, 6` for `0->2`), giving the standard decomposition of the PSF
- **CIFRA** — explicit state-by-state sum over 1⁺ intermediate states, taking a
  list of excitation energies and the corresponding nuclear matrix elements.
  This is the route to spectral shapes that are sensitive to the nuclear
  structure input, e.g. single-state dominance tests

The inner neutrino-energy integrands for all three sit in
`spades/spectra/{closure,taylor,cifra}_helpers.py`, JIT-compiled with numba —
the outer spectrum is a nested quadrature, so the innermost kernel is the whole
runtime.

**Q-values.** Taken from bundled tabulations (`data/mass_difference/`): AME2020 +
ENSDF2025, or the older KI 2012/2013 set, selectable per run. Runs abort early on
a negative Q-value rather than producing a plausible-looking wrong number.

## Installation

Requirements:

- Python ≥ 3.12 (the code uses PEP 695 `type` statements)
- A Fortran compiler (gfortran or ifort) and CMake ≥ 3.15
- NumPy, SciPy, numba, PyYAML, attrs, matplotlib, tqdm, hepunits, mpmath,
  periodictable, pyhepmc (resolved automatically)

The DHFS/RADIAL Fortran sources are compiled into shared libraries that are
loaded at runtime via `ctypes`, relative to the package, from `build/`. Install
in editable mode from the repository root so that the libraries and the Python
package stay in the same tree:

```bash
git clone git@github.com:stealex/2nubbspectra.git
cd 2nubbspectra
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

This produces `build/libradial.so` and `build/libdhfs.so` and installs the
`compute-spectra-psfs` entry point. A conda environment file (`2nubb.yml`) is
provided if you prefer conda-forge compilers.

Verify the atomic-structure layer alone:

```bash
python examples/simple_dhfs.py
```

## Quick start

Everything is driven by a single YAML file. Start from
`example_configs/82Se_2nu_2betaMinus_cifra.yaml` or one of the templates in
`config_files/`, which are commented option-by-option.

```bash
compute-spectra-psfs my_config.yaml \
    --energy_unit MeV \
    --distance_unit bohr_radius \
    --qvalues_file data/mass_difference/deltaM_AME2020_ENSDF2025.yaml \
    --verbose 2
```

CLI options:

| Flag | Meaning | Default |
|---|---|---|
| `--energy_unit` | any CLHEP energy unit, or `electron_mass`, `hartree_energy` | `MeV` |
| `--distance_unit` | any CLHEP distance unit, or `bohr_radius` | `fm` |
| `--qvalues_file` | YAML with `ANuc: Qval` entries, in MeV | KI 2012/2013 set |
| `--compute_2d_spectrum` | also produce the 2D spectrum | off |
| `--verbose` | 0 (quiet) … 5 (debug) | 0 |

Units given on the command line apply to the numbers in the config file — the
config itself is unit-agnostic.

## Configuration reference

```yaml
task: 82Se_2nu_2betaMinus            # label, printing only

process:
  name: 2nu_2betaMinus
  transition: 0->0                   # 0->0 | 0->2 | 0->02

initial_atom:
  name: 82Se                         # <A><Symbol>
  weight: -1.                        # negative -> use A
  electron_config: auto              # or a DHFS-style config file

bound_states:                        # DHFS + Dirac bound spectrum
  max_r: 100
  n_radial_points: 1000
  n_values: auto                     # auto | int | (nmin,nmax) | [n1,n2,...]
  k_values: auto                     # auto | int | (kmin,kmax) | {n: [k,...]}

scattering_states:                   # continuum states of the daughter ion
  max_r: 100
  n_radial_points: 1000
  n_ke_points: 100
  k_values: [-1, 1]

spectra_computation:
  method:
    name: Closure                    # Closure | Taylor | CIFRA
    enei: auto                       # Closure:  auto | float (MeV)
    # orders: [0, 2, 22, 4]          # Taylor
    # e1plus: [...]                  # CIFRA: 1+ excitation energies (MeV)
    # melems: [...]                  # CIFRA: matching matrix elements
  wave_function_evaluation: OnSurface
  nuclear_radius: auto               # auto -> 1.2·A^(1/3) fm
  types: [Single, Sum, Angular]
  energy_grid_type: lin              # lin | log
  corrections: [Exchange, Radiative]
  fermi_function_types: [Numeric]    # Numeric | PointLike | ChargedSphere
  total_ke: auto                     # auto -> from the Q-value file
  min_ke: 0.00001
  n_ke_points: 100
  compute_2d: False
  n_points_log_2d: 100               # log part of the 2D grid, up to e_max_log_2d
  e_max_log_2d: 0.1
  n_points_lin_2d: 100               # linear part, up to the Q-value

output:
  location: 82Se_2nu_2betaMinus_cifra
  what: [psfs, spectra]              # also: bound wavefunctions,
                                     # scattering wavefunctions, binding energies
  file_prefix: 82Se_2nu_2betaMinus_cifra
```

`fermi_function_types` and Taylor `orders` are lists: several variants are
computed in one run, share the same wave functions, and land side by side in the
output. That is the intended way to produce correction-by-correction comparisons.

## Output format

Two JSON files per run, in `output.location`:

**`spectra.json`**

```jsonc
{
  "energy_points": [...],            // MeV
  "Spectra": {
    "Numeric":   { "1": [...], "2": [...], "3": [...] },
    "Numeric_0": { ... }             // Taylor: one block per order
  },
  "PSFs": {
    "Numeric": { "1": G_single, "2": G_sum, "3": H }   // 1/year
  }
}
```

Spectrum keys are `SpectrumTypes`: `1` = single, `2` = summed, `3` = angular,
`4` = alpha. Outer keys are the Fermi-function type, suffixed with the Taylor
order where applicable.

**`spectra.json_2D.json`** — `e1_grid_2D`, `e2_grid_2D` and `Spectra_2D`, same
key scheme, written whenever `compute_2d` is set.

Plotting helpers for the common comparisons are in `plot_spectra.py`
(closure vs. Taylor, single/summed/angular, α coefficient) and `print_hls.py`
(CIFRA half-lives against closure and Taylor).

## Implementation notes

The pipeline is `DHFS.f` → bound wave functions + self-consistent potential →
RADIAL → scattering wave functions + phase shifts → Fermi functions
`f₀(ε)`, `f₁(ε)` → PSF and spectrum integrals. A few choices along that path are
worth knowing about before modifying the code:

- **Fortran interface via `ctypes`, not f2py** — `libdhfs.so`/`libradial.so` are
  loaded at runtime with explicit `argtypes`/`restype` declarations and Fortran
  name mangling (`func_`); 2D arrays cross the boundary as `order="F"`. This
  keeps the Fortran sources untouched and buildable by CMake on its own.
- **numba `@njit` on the neutrino integrands** — the ⟨K_N⟩/⟨L_N⟩ kernels and
  integrand assembly are evaluated O(10⁶) times per spectrum grid point, and are
  the only part of the calculation that needed to be compiled.
- **Adaptive quadrature** — `scipy.integrate.quad` (Gauss-Kronrod, DQAGSE) for
  the nested integrals, with variable limits supplied as callables and error
  tolerances enforced per sub-interval, rather than a fixed grid.
- **`@lru_cache` on Fermi-function and electron-integrand evaluations**, since
  the outer integration revisits the same (ε₁, ε₂) pairs across spectrum types.
- **`mpmath.hyp1f1`** for the point-like Fermi function: the Kummer ₁F₁ loses
  precision to cancellation at large |η| in double precision.
- **`scipy.special.spence`** (dilogarithm) for the radiative correction, Coulomb
  phase shifts from `scipy.special.gamma`, `CubicSpline`/`Akima1DInterpolator`
  for wave-function and correction interpolation.

## Batch and cluster runs

- **`run_all_nuclei.sh`** — sweeps a template config over an isotope list,
  substituting `{PARENT}`, `{TRANSITION}` and `{DEST}`, one output directory per
  nucleus. Isotope lists for all seven processes are kept in the script.
- **`scripts/gen_cifra_configs.py`** — generates cumulative CIFRA configs
  containing the first *N* intermediate states, for *N* = 1…*N*max. This is the
  convergence study: how many 1⁺ states does the spectrum actually need?
- **`scripts/submit_cifra_jobs.sh`** + **`scripts/cifra_job.sh`** — one SLURM job
  per config; the job stages work in `/dev/shm` and copies results back.

## Repository layout

```
spades/
  bin/compute_spectra_psfs.py   CLI: config -> atoms -> wave functions -> spectra -> JSON
  config.py                     YAML schema, unit conversion, "auto" resolution
  dhfs.py, dhfs_wrapper.py      DHFS self-consistent atomic potential (+ ctypes bridge)
  radial/                       RADIAL & DHFS Fortran sources, CMake build
  radial_wrapper.py             ctypes bridge to the Dirac radial solver
  wavefunctions.py              bound and scattering state handlers
  fermi_functions.py            PointLike / ChargedSphere / Numeric
  exchange.py                   exchange correction as a wave-function transformation
  spectra/
    base.py                     SpectrumBase / BetaSpectrumBase abstractions
    twobeta.py, ecbeta.py       ββ and ECβ⁺ spectra, per method
    twoec.py                    ECEC
    *_helpers.py                numba kernels for the neutrino integrands
    spectrum_writer.py          JSON serialisation
  ph.py                         constants, enums, unit handling
data/
  atomic_gs_configurations/     ground-state configurations, Z = 1…~100
  mass_difference/              Q-value tabulations (AME2020+ENSDF2025, KI)
config_files/, example_configs/ templates and ready-made configs
doc/                            the underlying literature
notebooks/                      analysis notebooks, LaTeX tables for publication
```

New process/method combinations are added by writing a factory and registering
it in `_SPECTRUM_REGISTRY` in `spades/bin/compute_spectra_psfs.py`, keyed by
`(ProcessTypes, method_name)`; the driver, the grids and the output layer need no
changes.

## Units and conventions

Internally everything is fm and MeV (via `hepunits`); the Fortran layer works in
atomic units and conversions are confined to the wrappers. PSFs are reported in
1/year. `--energy_unit`/`--distance_unit` control only how the config file is
interpreted, which is why the same config can be reused across unit conventions.

## Status and known limitations

- The unit tests under `tests/` predate the `src` → `spades` package rename and
  do not import as-is; they are kept because their physics checks (phase shifts,
  Fermi functions, exchange, DHFS) are still the right ones to port.
- `generate.py`, the Monte Carlo event sampler that draws (ε₁, ε₂) pairs from
  the 2D spectra for detector simulations, is a prototype and imports the same
  pre-rename layout.
- 0ν modes are implemented for light-neutrino exchange only.
- `Taylor` is available for 2νββ and 2νECEC; ECβ⁺ is closure-only.
- HDF5 output is enumerated in `ph.OutputFormatTypes` but only JSON is written.

## Citing

If you use SPADES, please cite the software DOI and the accompanying paper:

> S. Ghinescu and S. Stoica, *Updated results for kinematic factors in double
> beta decays*, Eur. Phys. J. C **86**, 496 (2026).
> [doi:10.1140/epjc/s10052-026-15656-6](https://doi.org/10.1140/epjc/s10052-026-15656-6)

> Software: [doi:10.5281/zenodo.12607448](https://doi.org/10.5281/zenodo.12607448)

The paper contains updated PSF tables for a large set of DBD isotopes, computed
with this code in both the closure and Taylor approaches, together with a
breakdown of the individual atomic corrections. PSFs for isotopes not tabulated
there are available on request — or reproducible directly with the configs in
this repository.

## References

The DBD formalism papers this implementation follows — Doi & Kotani, Haxton &
Stephenson, Tomoda, Suhonen & Civitarese, Kotila & Iachello, Stoica & Mirea — are
collected in `doc/`.

The radial solver is the RADIAL package (2018 version); its references are in the
header of `spades/radial/radial_new.f`:

> F. Salvat and R. Mayol, *Comput. Phys. Commun.* **62** (1991) 65;
> F. Salvat, J. M. Fernández-Varea and W. Williamson Jr., *Comput. Phys. Commun.*
> **90** (1995) 151.

The exchange correction follows the wave-function transformation of Nitescu et al.

## License

GPL-3.0 — see [LICENSE](LICENSE).

Developed at the International Centre for Advanced Training and Research in
Physics (IFIN-HH), Măgurele, Romania. User documentation contributed by
A. Dragomir.
