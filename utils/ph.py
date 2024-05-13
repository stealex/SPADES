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
CLOSUREMETHOD = 1
TAYLORMETHOD = 2

SINGLESPECTRUM = 1
SUMMEDSPECTRUM = 2
ANGULARSPECTRUM = 3

STANDARDPSF = 1
ANGULARPSF = 2
