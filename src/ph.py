# values taken from https://pdg.lbl.gov/2019/reviews/rpp2018-rev-phys-constants.pdf
from hepunits.units import *
from hepunits.constants import *
import os
import yaml
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

q_values_file = "qvalues_2nubb_KY13.yaml"


def read_qvalues(file_name: str):
    print(f"Reading {file_name}")
    if (file_name.startswith((".", "/"))):
        # we were given an absolute path
        file = open(file_name, 'r')
    else:
        # we were given a file name, check in the repo
        _dir_name = os.path.dirname(__file__)
        file = open(os.path.join(_dir_name, f"../data/{file_name}"))
    qvalues = yaml.safe_load(file)
    return qvalues


def to_distance_units():
    return fm/user_distance_unit


fermi_coupling_constant = 1.1663787E-5/(GeV**2)
v_ud = 0.97373

verbose = 0


POINTLIKEFERMIFUNCTIONS = 1
CHARGEDSPHEREFERMIFUNCTIONS = 2
NUMERICFERMIFUNCTIONS = 3
FERMIFUNCTIONS = {"PointLike": POINTLIKEFERMIFUNCTIONS,
                  "ChargedSphere": CHARGEDSPHEREFERMIFUNCTIONS,
                  "Numeric": NUMERICFERMIFUNCTIONS}

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

EXCHANGECORRECTION = 1
RADIATIVECORRECTION = 2
CORRECTIONS = {"Exchange": EXCHANGECORRECTION,
               "Radiative": RADIATIVECORRECTION}

ONSURFACE = 1
WEIGHTED = 2
WAVEFUNCTIONEVALUATION = {"OnSurface": ONSURFACE,
                          "Weighted": WEIGHTED}
