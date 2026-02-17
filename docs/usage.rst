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
