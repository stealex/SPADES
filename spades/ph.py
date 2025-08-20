# values taken from https://pdg.lbl.gov/2019/reviews/rpp2018-rev-phys-constants.pdf
from hepunits.units import *
from hepunits.constants import *
import os
import yaml
import logging
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


def read_mass_difference(file_name: str):
    print(f"Reading {file_name}")
    if (file_name.startswith((".", "/"))):
        # we were given an absolute path
        file = open(file_name, 'r')
    else:
        # we were given a file name, check in the repo
        _dir_name = os.path.dirname(__file__)
        file = open(os.path.join(
            _dir_name, f"../data/mass_difference/{file_name}"))
    qvalues = yaml.safe_load(file)
    return qvalues


def to_distance_units():
    return fm/user_distance_unit


fermi_coupling_constant = 1.1663787E-5/(GeV**2)
v_ud = 0.97373

verbose = 0


class ProcessTypes(IntEnum):
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
    CLOSUREMETHOD = 1
    TAYLORMETHOD = 2


SPECTRUM_METHODS = {"Closure": SpectrumMethod.CLOSUREMETHOD,
                    "Taylor": SpectrumMethod.TAYLORMETHOD}


class SpectrumTypes(IntEnum):
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
    EXCHANGE_CORRECTION = 1
    RADIATIVE_CORRECTION = 2


CORRECTIONS = {"Exchange": CorrectionTypes.EXCHANGE_CORRECTION,
               "Radiative": CorrectionTypes.RADIATIVE_CORRECTION}


class WFEvaluationTypes(IntEnum):
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
    JSONFORMAT = 1
    HDF5FORMAT = 2


OUTPUTFILEFORMAT = {"json": OutputFormatTypes.JSONFORMAT,
                    "hdf5": OutputFormatTypes.HDF5FORMAT}
