# values taken from https://pdg.lbl.gov/2019/reviews/rpp2018-rev-phys-constants.pdf
from hepunits.units import *
from hepunits.constants import *
import os
import yaml
import logging
electron_mass = 0.51099895000*MeV
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

delta_m_files = "deltaM_KY13.yaml"


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


TWONEUTRINO_TWOBMINUS = 1
NEUTRINOLESS_TWOBMINUS = 2
TWONEUTRINO_TWOBPLUS = 3
NEUTRINOLESS_TWOBPLUS = 4
TWONEUTRINO_BPLUSEC = 5
NEUTRINOLESS_BPLUSEC = 6
TWONEUTRINO_TWOEC = 7
PROCESSES = {"2nu_2betaMinus": TWONEUTRINO_TWOBMINUS,
             "0nu_2betaMinus": NEUTRINOLESS_TWOBMINUS,
             "2nu_2betaPlus": TWONEUTRINO_TWOBPLUS,
             "0nu_2betaPlus": NEUTRINOLESS_TWOBPLUS,
             "2nu_ECbetaPlus": TWONEUTRINO_BPLUSEC,
             "0nu_ECbetaPlus": NEUTRINOLESS_BPLUSEC,
             "2nu_2EC": TWONEUTRINO_TWOEC}

ZEROPLUS_TO_ZEROPLUS = 1
ZEROPLUS_TO_TWOPLUS = 2
TRANSITION_NAMES = {"0->0": ZEROPLUS_TO_ZEROPLUS,
                    "0->2": ZEROPLUS_TO_TWOPLUS}
TRANSITION_NAMES_REV = {ZEROPLUS_TO_ZEROPLUS: "0->0",
                        ZEROPLUS_TO_TWOPLUS: "0->2"}

LIGHT_NEUTRINO_EXCHANGE = 1
NEUTRINOLESS_MECHANISMS = {"LNE": LIGHT_NEUTRINO_EXCHANGE}
NEUTRINOLESS_MECHANISMS_REV = {LIGHT_NEUTRINO_EXCHANGE: "LNE"}


PROCESS_IONISATION = {TWONEUTRINO_TWOBMINUS: 2,
                      NEUTRINOLESS_TWOBMINUS: 2,
                      TWONEUTRINO_BPLUSEC: -1,
                      NEUTRINOLESS_BPLUSEC: -1,
                      TWONEUTRINO_TWOBPLUS: -2,
                      NEUTRINOLESS_TWOBPLUS: -2,
                      TWONEUTRINO_TWOEC: 0}


POINTLIKE_FERMIFUNCTIONS = 1
CHARGEDSPHERE_FERMIFUNCTIONS = 2
NUMERIC_FERMIFUNCTIONS = 3
FERMIFUNCTIONS = {"PointLike": POINTLIKE_FERMIFUNCTIONS,
                  "ChargedSphere": CHARGEDSPHERE_FERMIFUNCTIONS,
                  "Numeric": NUMERIC_FERMIFUNCTIONS}
FERMIFUNCTIONS_REV = {POINTLIKE_FERMIFUNCTIONS: "PointLike",
                      CHARGEDSPHERE_FERMIFUNCTIONS: "ChargedSphere",
                      NUMERIC_FERMIFUNCTIONS: "Numeric"}

CLOSUREMETHOD = 1
TAYLORMETHOD = 2
SPECTRUM_METHODS = {"Closure": CLOSUREMETHOD,
                    "Taylor": TAYLORMETHOD}

SINGLESPECTRUM = 1
SUMMEDSPECTRUM = 2
ANGULARSPECTRUM = 3
ALPHASPECTRUM = 4
SPECTRUM_TYPES = {"Single": SINGLESPECTRUM,
                  "Sum": SUMMEDSPECTRUM,
                  "Angular": ANGULARSPECTRUM,
                  "Alpha": ALPHASPECTRUM}
SPECTRUM_TYPES_REV = {SINGLESPECTRUM: "Single",
                      SUMMEDSPECTRUM: "Sum",
                      ANGULARSPECTRUM: "Angular",
                      ALPHASPECTRUM: "Alpha"}

SPECTRUM_TYPES_NICE = {SINGLESPECTRUM: "dG/de",
                       SUMMEDSPECTRUM: "dG/dT",
                       ANGULARSPECTRUM: "dH/de",
                       ALPHASPECTRUM: "alpha"}
SPECTRUM_TYPES_LATEX = {SINGLESPECTRUM: r"$\frac{1}{G}\frac{dG}{d\epsilon_1}$",
                        SUMMEDSPECTRUM: r"$\frac{1}{G}\frac{dG}{dT}$",
                        ANGULARSPECTRUM: r"$\frac{1}{H}\frac{dH}{d\epsilon_1}$",
                        ALPHASPECTRUM: r"$\alpha(\epsilon_1)$"}

PSF_TYPES_NICE = {SINGLESPECTRUM: "G_single",
                  SUMMEDSPECTRUM: "G_sum",
                  ANGULARSPECTRUM: "H",
                  ALPHASPECTRUM: "K"}

EXCHANGE_CORRECTION = 1
RADIATIVE_CORRECTION = 2
CORRECTIONS = {"Exchange": EXCHANGE_CORRECTION,
               "Radiative": RADIATIVE_CORRECTION}

ONSURFACE = 1
WEIGHTED = 2
WAVEFUNCTIONEVALUATION = {"OnSurface": ONSURFACE,
                          "Weighted": WEIGHTED}

# PATHS
gs_configurations_path = os.path.join(os.path.dirname(
    __file__), "../data/atomic_gs_configurations")
q_values_path = os.path.join(os.path.dirname(
    __file__), "../data/mass_difference/deltaM_KY13.yaml")

JSONFORMAT = 1
HDF5FORMAT = 2
OUTPUTFILEFORMAT = {"json": JSONFORMAT,
                    "hdf5": HDF5FORMAT}
