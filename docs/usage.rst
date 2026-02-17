Usage
=====

CLI entrypoint
--------------

The installed command is:

.. code-block:: bash

   spades path/to/config.yaml

Equivalent module form:

.. code-block:: bash

   python3 -m spades.bin.compute_spectra_psfs path/to/config.yaml

CLI arguments
-------------

Required positional argument:

- ``config_file``: path to YAML configuration file.

Optional flags:

- ``--verbose``: integer in ``[0, 5]``.
- ``--energy_unit``: output/input energy unit symbol (default ``MeV``).
- ``--distance_unit``: distance unit symbol (default ``fm``).
- ``--qvalues_file``: alternate YAML file with mass-difference values.
- ``--compute_2d_spectrum``: enable 2D spectra generation.

Exit behavior
-------------

The command raises Python exceptions for invalid config combinations (unknown process names, unsupported methods/options, or inconsistent physics settings). In CI, this is useful to catch bad configurations early.

Process-Specific Configuration
------------------------------

Use one template per process from ``config_files/``.

2nu: 2betaMinus (detailed example)
----------------------------------

Start from ``config_files/2nu_2betaMinus_template.yaml``.

Example:

.. code-block:: yaml

   task: 76Ge_2nu_2betaMinus

   process:
     name: 2nu_2betaMinus         # process channel
     transition: 0->0             # allowed: 0->0, 0->2

   initial_atom:
     name: 76Ge                   # isotope label <A><Symbol>
     weight: -1.0                 # negative => use A as atomic weight
     electron_config: auto        # auto or custom YAML electron config file

   bound_states:
     max_r: 100                   # radial box size (distance unit from CLI)
     n_radial_points: 10000       # radial grid size for bound-state solver
     n_values: auto               # principal quantum numbers (auto/manual)
     k_values: auto               # Dirac kappa values (auto/manual)

   scattering_states:
     max_r: 100                   # radial box size for scattering states
     n_radial_points: 10000       # radial grid points for scattering states
     n_ke_points: 100             # scattering energies used to tabulate waves
     k_values: [-1, 1]            # scattering kappa channels

   spectra_computation:
     method:
       name: Closure              # Closure or Taylor
       enei: auto                 # closure energy (float or auto)
     wave_function_evaluation: OnSurface   # OnSurface or Weighted
     nuclear_radius: auto         # auto or positive float
     types: [Single, Angular]     # requested spectra: Single, Sum, Angular
     energy_grid_type: lin        # lin or log
     corrections: [Exchange, Radiative]    # optional corrections
     fermi_function_types: [Numeric]       # Numeric, PointLike, ChargedSphere
     total_ke: auto               # float or auto from mass-difference table
     min_ke: 1.0e-5               # lower kinetic-energy bound
     n_ke_points: 100             # number of points in 1D spectra grid
     compute_2d: false            # whether to also compute 2D spectra
     n_points_log_2d: 100         # 2D grid: logarithmic segment points
     e_max_log_2d: 0.1            # 2D grid: end of logarithmic segment
     n_points_lin_2d: 100         # 2D grid: linear segment points

   output:
     location: out_76Ge_2nu_2betaMinus     # destination directory
     what: [spectra, psfs]                 # output groups to write
     file_prefix: 76Ge_2nu_2betaMinus      # output naming prefix

Notes:

- ``--energy_unit`` and ``--distance_unit`` CLI flags define the units expected by config numeric fields.
- ``total_ke: auto`` and some transition-dependent quantities use the file from ``--qvalues_file``.

2nu: 2betaPlus
--------------

Use ``config_files/2nu_2betaPlus_template.yaml``.
For option meanings, follow the detailed ``2nu_2betaMinus`` subsection above.

2nu: ECbetaPlus
---------------

Use ``config_files/2nu_ECbetaPlus_template.yaml``.
For option meanings, follow the detailed ``2nu_2betaMinus`` subsection above.

2nu: 2EC
--------

Use ``config_files/2nu_2EC_template.yaml``.
For option meanings, follow the detailed ``2nu_2betaMinus`` subsection above.

0nu: 2betaMinus
---------------

Use ``config_files/0nu_2betaMinus_template.yaml``.
For option meanings, follow the detailed ``2nu_2betaMinus`` subsection above.
0nu templates additionally include ``process.mechanism`` (currently ``LNE``).

0nu: 2betaPlus
--------------

Use ``config_files/0nu_2betaPlus_template.yaml``.
For option meanings, follow the detailed ``2nu_2betaMinus`` subsection above.
0nu templates additionally include ``process.mechanism`` (currently ``LNE``).

0nu: ECbetaPlus
---------------

Use ``config_files/0nu_ECbetaPlus_template.yaml``.
For option meanings, follow the detailed ``2nu_2betaMinus`` subsection above.
0nu templates additionally include ``process.mechanism`` (currently ``LNE``).
