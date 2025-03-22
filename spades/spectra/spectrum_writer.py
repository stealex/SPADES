from spades.spectra.base import SpectrumBase
from spades import ph
import json
import numpy as np


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class SpectrumWriter:
    def __init__(self, format: str = "json", write_spectra=True, write_psfs=True) -> None:
        self.spectra_to_write = {}
        self.format = ph.OUTPUTFILEFORMAT[format]
        self.write_spectra = write_spectra
        self.wrtie_psfs = write_psfs

    def add_spectrum(self, spectrum: SpectrumBase, name: str):
        if (name in self.spectra_to_write):
            raise ValueError(f"key {name} already present")
        self.spectra_to_write[name] = spectrum
        # check if we have energy_grid
        if (getattr(spectrum, "energy_points", None) is None):
            self.has_energy_grid = False
        else:
            self.has_energy_grid = True

        # check if we have 2D
        if (getattr(spectrum, "e1_grid_2D", None) is None):
            self.has_2D = False
        else:
            self.has_2D = True

    def write(self, file_name: str):
        if (self.format == ph.JSONFORMAT):
            self.write_json(file_name)

    def write_json(self, file_name: str):
        output_structure = {}
        output_structure_2D = {}
        some_key = next(iter(self.spectra_to_write))
        output_structure["energy_points"] = getattr(
            self.spectra_to_write[some_key], "energy_points", None)
        output_structure["Spectra"] = {}
        output_structure["PSFs"] = {}

        output_structure_2D["e1_grid_2D"] = getattr(
            self.spectra_to_write[some_key], "e1_grid_2D", None)
        output_structure_2D["e2_grid_2D"] = getattr(
            self.spectra_to_write[some_key], "e2_grid_2D", None)
        output_structure_2D["Spectra_2D"] = {}

        for key in self.spectra_to_write:
            if (self.write_spectra):
                output_structure["Spectra"][key] = getattr(
                    self.spectra_to_write[key], "spectrum_values", None)
                if (self.has_2D):
                    output_structure_2D["Spectra_2D"][key] = getattr(
                        self.spectra_to_write[key], "spectrum_2D_values", None)

            if (self.wrtie_psfs):
                output_structure["PSFs"][key] = self.spectra_to_write[key].psfs

        with open(file_name, 'w') as f:
            json.dump(output_structure, f, ensure_ascii=False,
                      indent=4, cls=NumpyArrayEncoder)
        with open(f"{file_name}_2D.json", 'w') as f:
            json.dump(output_structure_2D, f, ensure_ascii=False,
                      indent=4, cls=NumpyArrayEncoder)
