"""Legacy output configuration container."""

class output_config:
    """Backwards-compatible output switches container."""

    def __init__(self, 
                 location,
                 out_bound_wavefunctions=False,
                 out_scattering_wavefunctions=False,
                 out_spectra=False,
                 out_psfs=False,
                 out_binding_energies=False) -> None:
        """Initialize legacy output switches.

        Parameters
        ----------
        location:
            Output directory path.
        out_bound_wavefunctions, out_scattering_wavefunctions, out_spectra, out_psfs, out_binding_energies:
            Boolean toggles controlling which artifacts are written.
        """
        self.location = location
        self.out_bound_wavefunctions = out_bound_wavefunctions
        self.out_scattering_wavefunctions = out_scattering_wavefunctions
        self.out_spectra = out_spectra
        self.out_psfs = out_psfs
        self.out_binding_energies = out_binding_energies
        
