Overview
========

SPADES computes spectra and phase-space factors (PSFs) for double-beta decay channels.

Computation layers
------------------

SPADES is organized around four layers:

1. Atomic and wavefunction solvers (``spades.dhfs``, ``spades.wavefunctions``)
2. Fermi-function models (``spades.fermi_functions``)
3. Spectra/PSF engines (``spades.spectra``)
4. Workflow and IO orchestration (``spades.config``, ``spades.io_handler``, ``spades.bin.compute_spectra_psfs``)

Supported process names
-----------------------

Use the ``process.name`` field in YAML with one of:

- ``2nu_2betaMinus``
- ``0nu_2betaMinus``
- ``2nu_2betaPlus``
- ``0nu_2betaPlus``
- ``2nu_ECbetaPlus``
- ``0nu_ECbetaPlus``
- ``2nu_2EC``

Available methods and models
----------------------------

- Spectrum methods: ``Closure``, ``Taylor``
- Fermi-function backends: ``Numeric``, ``PointLike``, ``ChargedSphere``
- Optional corrections: ``Exchange``, ``Radiative``

Typical workflow
----------------

1. Prepare a YAML config (start from ``config_files/*_template.yaml``).
2. Run the CLI: ``spades path/to/config.yaml``.
3. Read produced outputs (typically ``spectra.json``) from your output directory.
