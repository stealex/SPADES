Overview
========

SPADES is organized around four computational layers:

1. Atomic and wavefunction solvers (`spades.dhfs`, `spades.wavefunctions`).
2. Coulomb/Fermi-function modeling (`spades.fermi_functions`).
3. Spectrum and phase-space-factor engines (`spades.spectra`).
4. Input/output and workflow orchestration (`spades.config`, `spades.io_handler`, `spades.bin.compute_spectra_psfs`).

The code is designed to support multiple decay channels and approximation schemes while preserving reproducible text/binary outputs for post-processing.
