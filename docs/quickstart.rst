Quickstart
==========

Run a first calculation
-----------------------

1. Copy a template from ``config_files/``.
2. Replace placeholders (for example ``{PARENT}``, ``{TRANSITION}``, ``{DEST}``).
3. Run the CLI.

Example minimal config (2nu 2beta-)
------------------------------------

.. code-block:: yaml

   task: 76Ge_2nu_2betaMinus

   process:
     name: 2nu_2betaMinus
     transition: 0->0

   initial_atom:
     name: 76Ge
     weight: -1.0
     electron_config: auto

   bound_states:
     max_r: 100
     n_radial_points: 5000
     n_values: auto
     k_values: auto

   scattering_states:
     max_r: 100
     n_radial_points: 5000
     n_ke_points: 120
     k_values: [-1, 1]

   spectra_computation:
     method:
       name: Closure
       enei: auto
     wave_function_evaluation: OnSurface
     nuclear_radius: auto
     types: [Single, Angular]
     energy_grid_type: lin
     corrections: [Exchange, Radiative]
     fermi_function_types: [Numeric]
     total_ke: auto
     min_ke: 1.0e-5
     n_ke_points: 200
     compute_2d: false
     n_points_log_2d: 100
     e_max_log_2d: 0.1
     n_points_lin_2d: 100

   output:
     location: out_76Ge
     what: [spectra, psfs]
     file_prefix: 76Ge_2nu_2betaMinus

Run command
-----------

.. code-block:: bash

   spades path/to/config.yaml

You can also invoke the module directly:

.. code-block:: bash

   python3 -m spades.bin.compute_spectra_psfs path/to/config.yaml

Useful CLI options
------------------

- ``--verbose {0..5}``
- ``--energy_unit MeV``
- ``--distance_unit fm``
- ``--qvalues_file data/mass_difference/deltaM_KI_2012_2013.yaml``
- ``--compute_2d_spectrum``
