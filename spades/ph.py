"""Physical constants, units, enums, and name maps used across SPADES."""

import os

# values taken from https://pdg.lbl.gov/2019/reviews/rpp2018-rev-phys-constants.pdf
try:
    from hepunits.units import *
    from hepunits.constants import *
except ModuleNotFoundError as exc:
    if os.environ.get("SPADES_ALLOW_UNIT_FALLBACK", "0") != "1":
        raise ModuleNotFoundError(
            "Missing dependency 'hepunits'. Install SPADES runtime dependencies "
            "(e.g. `pip install -e .`) or set SPADES_ALLOW_UNIT_FALLBACK=1 "
            "for documentation-only builds."
        ) from exc
    # Fallback units for documentation/static workflows where hepunits is absent.
    eV = 1.0
    keV = 1.0e3 * eV
    MeV = 1.0e6 * eV
    GeV = 1.0e9 * eV
    second = 1.0
    year = 365.25 * 24.0 * 3600.0 * second
    meter = 1.0
    angstrom = 1.0e-10 * meter
    fermi = 1.0e-15 * meter
    fm = fermi
    hbarc = 197.3269804 * MeV * fm
import yaml
import logging
from dataclasses import dataclass
from enum import IntEnum
electron_mass = 0.51099895000*MeV
proton_mass = 938.27208943*MeV
hartree_energy = 2.0*13.605693122994*eV
bohr_radius = 0.529177210903*angstrom
fine_structure = 1/137.035999084
hc = hbarc/(MeV*fermi)


user_distance_unit = fm
user_energy_unit = MeV
user_psf_unit = 1/year
user_distance_unit_name = "fm"
user_energy_unit_name = "MeV"
user_psf_unit_name = "1/year"

delta_m_files = "deltaM_KI_2012_2013.yaml"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeSettings:
    """Mutable-at-runtime settings used by CLI workflows.

    Parameters
    ----------
    verbose:
        Verbosity level in ``[0, 5]``.
    distance_unit_name:
        Symbolic distance unit name (must be defined in :mod:`spades.ph`).
    energy_unit_name:
        Symbolic energy unit name (must be defined in :mod:`spades.ph`).
    qvalues_file:
        Mass-difference YAML file name or absolute path.
    """

    verbose: int = 0
    distance_unit_name: str = "fm"
    energy_unit_name: str = "MeV"
    qvalues_file: str = delta_m_files


def _resolve_unit(name: str):
    """Resolve a unit symbol from this module namespace.

    Parameters
    ----------
    name:
        Unit symbol (for example ``"MeV"`` or ``"fm"``).

    Returns
    -------
    float
        Numeric unit scale.
    """
    if name not in globals():
        raise ValueError(f"Unknown unit '{name}'")
    return globals()[name]


def apply_runtime_settings(settings: RuntimeSettings) -> None:
    """Apply runtime settings to module-level variables used by legacy APIs.

    Parameters
    ----------
    settings:
        Runtime settings selected by the CLI.
    """
    global verbose, user_distance_unit_name, user_energy_unit_name
    global user_distance_unit, user_energy_unit, delta_m_files
    verbose = settings.verbose
    user_distance_unit_name = settings.distance_unit_name
    user_energy_unit_name = settings.energy_unit_name
    user_distance_unit = _resolve_unit(settings.distance_unit_name)
    user_energy_unit = _resolve_unit(settings.energy_unit_name)
    delta_m_files = settings.qvalues_file


def read_mass_difference(file_name: str):
    """Load tabulated mass differences from YAML.

    Parameters
    ----------
    file_name:
        Relative file name inside ``data/mass_difference`` or an absolute path.

    Returns
    -------
    dict
        Parsed YAML mapping with isotope keys and mass-difference metadata.
    """
    logger.info("Reading mass differences from %s", file_name)
    if (file_name.startswith((".", "/"))):
        # we were given an absolute path
        with open(file_name, 'r') as file:
            qvalues = yaml.safe_load(file)
    else:
        # we were given a file name, check in the repo
        _dir_name = os.path.dirname(__file__)
        with open(os.path.join(_dir_name, f"../data/mass_difference/{file_name}"), 'r') as file:
            qvalues = yaml.safe_load(file)
    return qvalues


def to_distance_units():
    """Return conversion factor from internal ``fm`` to ``user_distance_unit``."""
    return fm/user_distance_unit


fermi_coupling_constant = 1.1663787E-5/(GeV**2)
v_ud = 0.97373

verbose = 0


class ProcessTypes(IntEnum):
    """Supported double-beta decay channels."""

    TWONEUTRINO_TWOBMINUS = 1
    NEUTRINOLESS_TWOBMINUS = 2
    TWONEUTRINO_TWOBPLUS = 3
    NEUTRINOLESS_TWOBPLUS = 4
    TWONEUTRINO_BPLUSEC = 5
    NEUTRINOLESS_BPLUSEC = 6
    TWONEUTRINO_TWOEC = 7


PROCESS_NAMES_MAP = {"2nu_2betaMinus": ProcessTypes.TWONEUTRINO_TWOBMINUS,
                     "0nu_2betaMinus": ProcessTypes.NEUTRINOLESS_TWOBMINUS,
                     "2nu_2betaPlus": ProcessTypes.TWONEUTRINO_TWOBPLUS,
                     "0nu_2betaPlus": ProcessTypes.NEUTRINOLESS_TWOBPLUS,
                     "2nu_ECbetaPlus": ProcessTypes.TWONEUTRINO_BPLUSEC,
                     "0nu_ECbetaPlus": ProcessTypes.NEUTRINOLESS_BPLUSEC,
                     "2nu_2EC": ProcessTypes.TWONEUTRINO_TWOEC}


class TransitionTypes(IntEnum):
    """Nuclear transition channels between initial and final states."""

    ZEROPLUS_TO_ZEROPLUS = 1
    ZEROPLUS_TO_ZEROTWOPLUS = 2
    ZEROPLUS_TO_TWOPLUS = 3


TRANSITION_NAMES_MAP = {"0->0": TransitionTypes.ZEROPLUS_TO_ZEROPLUS,
                        "0->02": TransitionTypes.ZEROPLUS_TO_ZEROTWOPLUS,
                        "0->2": TransitionTypes.ZEROPLUS_TO_TWOPLUS}
TRANSITION_NAMES_MAP_REV = {TransitionTypes.ZEROPLUS_TO_ZEROPLUS: "0->0",
                            TransitionTypes.ZEROPLUS_TO_ZEROTWOPLUS: "0->02",
                            TransitionTypes.ZEROPLUS_TO_TWOPLUS: "0->2"}


class TaylorOrders(IntEnum):
    """Orders used in the Taylor-expansion spectra method."""

    ZERO = 0
    TWO = 2
    TWOTWO = 22
    FOUR = 4
    SIX = 6


TAYLOR_ORDER_NAMES_MAP = {"0": TaylorOrders.ZERO,
                          "2": TaylorOrders.TWO,
                          "22": TaylorOrders.TWOTWO,
                          "4": TaylorOrders.FOUR,
                          "6": TaylorOrders.SIX}
TAYLOR_ORDER_NAMES_MAP_REV = {TaylorOrders.ZERO: "0",
                              TaylorOrders.TWO: "2",
                              TaylorOrders.TWOTWO: "22",
                              TaylorOrders.FOUR: "4",
                              TaylorOrders.SIX: "6"}


class NeutrinoLessModes(IntEnum):
    """Neutrinoless mechanisms supported by SPADES."""

    LIGHT_NEUTRINO_EXCHANGE = 1


NEUTRINOLESS_MECHANISMS_MAP = {
    "LNE": NeutrinoLessModes.LIGHT_NEUTRINO_EXCHANGE}
NEUTRINOLESS_MECHANISMS_MAP_REV = {
    NeutrinoLessModes.LIGHT_NEUTRINO_EXCHANGE: "LNE"}


PROCESS_IONISATION = {ProcessTypes.TWONEUTRINO_TWOBMINUS: 2,
                      ProcessTypes.NEUTRINOLESS_TWOBMINUS: 2,
                      ProcessTypes.TWONEUTRINO_BPLUSEC: -1,
                      ProcessTypes.NEUTRINOLESS_BPLUSEC: -1,
                      ProcessTypes.TWONEUTRINO_TWOBPLUS: -2,
                      ProcessTypes.NEUTRINOLESS_TWOBPLUS: -2,
                      ProcessTypes.TWONEUTRINO_TWOEC: 0}


class FermiFunctionTypes(IntEnum):
    """Fermi-function evaluation backends."""

    POINTLIKE_FERMIFUNCTIONS = 1
    CHARGEDSPHERE_FERMIFUNCTIONS = 2
    NUMERIC_FERMIFUNCTIONS = 3


FERMIFUNCTIONS_MAP = {"PointLike": FermiFunctionTypes.POINTLIKE_FERMIFUNCTIONS,
                      "ChargedSphere": FermiFunctionTypes.CHARGEDSPHERE_FERMIFUNCTIONS,
                      "Numeric": FermiFunctionTypes.NUMERIC_FERMIFUNCTIONS}
FERMIFUNCTIONS_MAP_REV = {FermiFunctionTypes.POINTLIKE_FERMIFUNCTIONS: "PointLike",
                          FermiFunctionTypes.CHARGEDSPHERE_FERMIFUNCTIONS: "ChargedSphere",
                          FermiFunctionTypes.NUMERIC_FERMIFUNCTIONS: "Numeric"}


class SpectrumMethod(IntEnum):
    """Spectrum-computation methods."""

    CLOSUREMETHOD = 1
    TAYLORMETHOD = 2


SPECTRUM_METHODS = {"Closure": SpectrumMethod.CLOSUREMETHOD,
                    "Taylor": SpectrumMethod.TAYLORMETHOD}


class SpectrumTypes(IntEnum):
    """Output spectra families."""

    SINGLESPECTRUM = 1
    SUMMEDSPECTRUM = 2
    ANGULARSPECTRUM = 3
    ALPHASPECTRUM = 4


SPECTRUM_TYPES = {"Single": SpectrumTypes.SINGLESPECTRUM,
                  "Sum": SpectrumTypes.SUMMEDSPECTRUM,
                  "Angular": SpectrumTypes.ANGULARSPECTRUM,
                  "Alpha": SpectrumTypes.ALPHASPECTRUM}
SPECTRUM_TYPES_REV = {SpectrumTypes.SINGLESPECTRUM: "Single",
                      SpectrumTypes.SUMMEDSPECTRUM: "Sum",
                      SpectrumTypes.ANGULARSPECTRUM: "Angular",
                      SpectrumTypes.ALPHASPECTRUM: "Alpha"}

SPECTRUM_TYPES_NICE = {SpectrumTypes.SINGLESPECTRUM: "dG/de",
                       SpectrumTypes.SUMMEDSPECTRUM: "dG/dT",
                       SpectrumTypes.ANGULARSPECTRUM: "dH/de",
                       SpectrumTypes.ALPHASPECTRUM: "alpha"}
SPECTRUM_TYPES_LATEX = {SpectrumTypes.SINGLESPECTRUM: r"$\frac{1}{G}\frac{dG}{d\epsilon_1}$",
                        SpectrumTypes.SUMMEDSPECTRUM: r"$\frac{1}{G}\frac{dG}{dT}$",
                        SpectrumTypes.ANGULARSPECTRUM: r"$\frac{1}{H}\frac{dH}{d\epsilon_1}$",
                        SpectrumTypes.ALPHASPECTRUM: r"$\alpha(\epsilon_1)$"}

PSF_TYPES_NICE = {SpectrumTypes.SINGLESPECTRUM: "G_single",
                  SpectrumTypes.SUMMEDSPECTRUM: "G_sum",
                  SpectrumTypes.ANGULARSPECTRUM: "H",
                  SpectrumTypes.ALPHASPECTRUM: "K"}


class CorrectionTypes(IntEnum):
    """Corrections that can be applied to spectra/PSFs."""

    EXCHANGE_CORRECTION = 1
    RADIATIVE_CORRECTION = 2


CORRECTIONS = {"Exchange": CorrectionTypes.EXCHANGE_CORRECTION,
               "Radiative": CorrectionTypes.RADIATIVE_CORRECTION}


class WFEvaluationTypes(IntEnum):
    """Wavefunction evaluation strategies at the nuclear scale."""

    ONSURFACE = 1
    WEIGHTED = 2


WAVEFUNCTIONEVALUATION = {"OnSurface": WFEvaluationTypes.ONSURFACE,
                          "Weighted": WFEvaluationTypes.WEIGHTED}

# PATHS
gs_configurations_path = os.path.join(os.path.dirname(
    __file__), "../data/atomic_gs_configurations")
q_values_path = os.path.join(os.path.dirname(
    __file__), "../data/mass_difference/deltaM_KY13.yaml")


class OutputFormatTypes(IntEnum):
    """Supported output serialization formats."""

    JSONFORMAT = 1
    HDF5FORMAT = 2


OUTPUTFILEFORMAT = {"json": OutputFormatTypes.JSONFORMAT,
                    "hdf5": OutputFormatTypes.HDF5FORMAT}
