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
    def __init__(self, format: str = "json") -> None:
        self.spectra_to_write = {}
        self.format = ph.OUTPUTFILEFORMAT[format]

    def add_spectrum(self, spectrum: SpectrumBase, name: str):
        if (name in self.spectra_to_write):
            raise ValueError(f"key {name} already present")
        self.spectra_to_write[name] = spectrum
        # check if we have energy_grid
        if (getattr(spectrum, "energy_points", None) is None):
            self.has_energy_grid = False
        else:
            self.has_energy_grid = True

    def write(self, file_name: str):
        if (self.format == ph.JSONFORMAT):
            self.write_json(file_name)

    def write_json(self, file_name: str):
        output_structure = {}
        some_key = next(iter(self.spectra_to_write))
        output_structure["energy_points"] = getattr(
            self.spectra_to_write[some_key], "energy_points", None)
        output_structure["Spectra"] = {}
        output_structure["PSFs"] = {}

        for key in self.spectra_to_write:
            output_structure["Spectra"][key] = getattr(
                self.spectra_to_write[key], "spectrum_values", None)
            output_structure["PSFs"][key] = self.spectra_to_write[key].psfs
        with open(file_name, 'w') as f:
            json.dump(output_structure, f, ensure_ascii=False,
                      indent=4, cls=NumpyArrayEncoder)
