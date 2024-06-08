class output_config:
    def __init__(self, 
                 location,
                 out_bound_wavefunctions=False,
                 out_scattering_wavefunctions=False,
                 out_spectra=False,
                 out_psfs=False,
                 out_binding_energies=False) -> None:
        self.location = location
        self.out_bound_wavefunctions = out_bound_wavefunctions
        self.out_scattering_wavefunctions = out_scattering_wavefunctions
        self.out_spectra = out_spectra
        self.out_psfs = out_psfs
        self.out_binding_energies = out_binding_energies
        