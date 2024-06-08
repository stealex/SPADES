# values taken from https://pdg.lbl.gov/2019/reviews/rpp2018-rev-phys-constants.pdf
# distance units
from hepunits.units import *
from hepunits.constants import *
electron_mass = 0.51099895000*MeV
hartree_energy = 2.0*13.605693122994*eV
bohr_radius = 0.529177210903*angstrom
fine_structure = 1/137.035999084
hc = hbarc/(MeV*fermi)

user_distance_unit = 0.0
user_energy_unit = 0.0
user_distance_unit_name = ""
user_energy_unit_name = ""

fermi_coupling_constant = 1.1663787E-5/(GeV**2)
v_ud = 0.97373

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

EXCHANGECORRECTION = 1
RADIATIVECORRECTION = 2
CORRECTIONS = {"Exchange": EXCHANGECORRECTION,
               "Radiative": RADIATIVECORRECTION}

ONSURFACE = 1
WEIGHTED = 2
WAVEFUNCTIONEVALUATION = {"OnSurface": ONSURFACE,
                          "Weighted": WEIGHTED}
