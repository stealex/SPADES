from handlers.wavefunction_handlers import dhfs_handler, bound_handler, scattering_handler
from configs.wavefunctions_config import atomic_system, radial_scattering_config


class wavefunctions_handler:
    def __init__(self, atom: atomic_system, bound_conf=None, scattering_conf: radial_scattering_config | None = None) -> None:
        self.atomic_system = atom
        if (bound_conf is None) and (scattering_conf is None):
            raise ValueError(
                "At least one of bound_conf or scattering_conf should be passed")
        if not (bound_conf is None):
            self.bound_config = bound_conf
        if not (scattering_conf is None):
            self.scattering_config = scattering_conf

    def run_dhfs(self) -> None:
        self.dhfs_handler = dhfs_handler(
            self.atomic_system, self.atomic_system.name)
        self.dhfs_handler.run_dhfs(self.bound_config.max_r,
                                   self.bound_config.n_radial_points,
                                   iverbose=0)

        self.dhfs_handler.retrieve_dhfs_results()
        self.dhfs_handler.build_modified_potential()

    def find_bound_states(self):
        self.bound_handler = bound_handler(self.atomic_system.Z,
                                           int(self.atomic_system.occ_values.sum()),
                                           self.bound_config)

        self.bound_handler.find_bound_states(self.dhfs_handler.rad_grid,
                                             self.dhfs_handler.rv_modified)

    def find_scattering_states(self):
        # solve scattering states in final atom
        self.scattering_handler = scattering_handler(self.atomic_system.Z,
                                                     int(self.atomic_system.occ_values.sum(
                                                     )),
                                                     self.scattering_config)
        self.scattering_handler.set_potential(
            self.dhfs_handler.rad_grid, self.dhfs_handler.rv_modified)

        self.scattering_handler.compute_scattering_states()

    def find_all_wavefunctions(self):
        try:
            self.run_dhfs()
            self.find_bound_states()
        except AttributeError:
            pass

        try:
            self.find_scattering_states()
        except AttributeError:
            pass
