Usage
=====

Command-line entry point
------------------------

SPADES includes a CLI entry script:

.. code-block:: bash

   python -m spades.bin.compute_spectra_psfs path/to/config.yaml

Configuration
-------------

Run configuration is parsed by :class:`spades.config.RunConfig`, which resolves:

- process channel and transition types,
- unit conversions to internal MeV/fm representation,
- bound/scattering wavefunction settings,
- spectra computation options and output selections.

Outputs
-------

Serialized data products are handled by :mod:`spades.io_handler` and include:

- bound/scattering wavefunctions,
- 1D and 2D spectra,
- phase-space factors (PSFs),
- tabulated Fermi functions.
