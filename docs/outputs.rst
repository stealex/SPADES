Outputs
=======

Primary output
--------------

The CLI writes a ``spectra.json`` file in the configured output directory when ``output`` is enabled and at least one of ``spectra`` or ``psfs`` is requested.

This file is produced through :class:`spades.spectra.spectrum_writer.SpectrumWriter`.

Content organization
--------------------

Results are grouped by Fermi-function backend (for example ``Numeric`` or ``PointLike``).
For Taylor runs, additional nesting by order is used.

Depending on configuration, outputs can include:

- 1D spectra arrays
- 2D spectra arrays (if ``compute_2d: true``)
- Integrated PSF values

Text/binary helpers
-------------------

The module :mod:`spades.io_handler` also contains helpers to serialize and load:

- Bound/scattering wavefunctions
- Spectra tables
- Fermi-function tables

These utilities are useful for post-processing workflows and interoperability with existing analysis scripts.

Reproducibility notes
---------------------

For reproducible runs, keep these fixed across comparisons:

- Same YAML configuration
- Same units/CLI overrides
- Same mass-difference file (``--qvalues_file``)
- Same SPADES version/commit
