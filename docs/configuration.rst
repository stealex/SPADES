Configuration
=============

Start point
-----------

Use one of the templates in ``config_files/`` as a base. The CLI expects a top-level YAML document with these sections:

- ``task``
- ``process``
- ``initial_atom``
- ``spectra_computation``
- Optional: ``bound_states``, ``scattering_states``, ``output``

process
-------

Fields:

- ``name`` (required): one of
  - ``2nu_2betaMinus``, ``0nu_2betaMinus``
  - ``2nu_2betaPlus``, ``0nu_2betaPlus``
  - ``2nu_ECbetaPlus``, ``0nu_ECbetaPlus``
  - ``2nu_2EC``
- ``transition`` (optional for many channels): ``0->0``, ``0->02``, ``0->2``
- ``mechanism`` (for 0nu modes): currently ``LNE``

initial_atom
------------

Common fields:

- ``name``: isotope label (example: ``76Ge``)
- ``weight``: use ``-1`` to default to mass number
- ``electron_config``: ``auto`` or path to a YAML shell configuration

bound_states
------------

Used to compute bound orbitals:

- ``max_r``
- ``n_radial_points``
- ``n_values`` (often ``auto``)
- ``k_values`` (often ``auto``)

scattering_states
-----------------

Used to compute scattering orbitals:

- ``max_r``
- ``n_radial_points``
- ``n_ke_points``
- ``k_values`` (for example ``[-1, 1]``)

``min_ke`` and ``max_ke`` are derived internally from ``spectra_computation``.

spectra_computation
-------------------

Main controls:

- ``method``
  - ``name``: ``Closure`` or ``Taylor``
  - For ``Closure``: ``enei`` as float or ``auto``
  - For ``Taylor``: ``orders`` list such as ``[0, 2, 22, 4]`` or ``auto``
- ``wave_function_evaluation``: ``OnSurface`` or ``Weighted``
- ``nuclear_radius``: positive float or ``auto``
- ``types``: list of ``Single``, ``Sum``, ``Angular``
- ``energy_grid_type``: ``lin`` or ``log``
- ``corrections``: list of ``Exchange``, ``Radiative``
- ``fermi_function_types``: list of ``Numeric``, ``PointLike``, ``ChargedSphere``
- ``total_ke``: positive float or ``auto``
- ``min_ke``, ``n_ke_points``
- 2D options: ``compute_2d``, ``n_points_log_2d``, ``e_max_log_2d``, ``n_points_lin_2d``

output
------

When provided, output writing is enabled.

Fields:

- ``location``: output directory
- ``what``: list of output groups. Valid values are:
  - ``spectra``
  - ``psfs``
  - ``bound wavefunctions``
  - ``scattering wavefunctions``
  - ``binding energies``
- ``file_prefix``: label prefix used in output naming
