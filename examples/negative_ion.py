from xml.sax import handler
from spades.dhfs import DHFSHandler, AtomicSystem, create_ion
from spades import ph

import matplotlib.pyplot as plt

ph.verbose = 1

tmp_atom = AtomicSystem(name="45Ca", electron_config="auto")

ref_atom = create_ion(tmp_atom, tmp_atom.Z-2)
z_ref = ref_atom.Z
print(f"z_ref = {z_ref}")

neutral_atom_ref = AtomicSystem(atomic_number=z_ref, mass_number=45)
handler_neutral_ref = DHFSHandler(neutral_atom_ref, "neutral ref")
handler_neutral_ref.run_dhfs(100, 1000)
handler_neutral_ref.retrieve_dhfs_results()

# create a positive ion with the same number of protons
neutral_atom_zp = AtomicSystem(atomic_number=z_ref-2, mass_number=43)
positive_ion = create_ion(neutral_atom_zp, z_ref)
handler_positive_ion = DHFSHandler(positive_ion, "positive ion")
handler_positive_ion.run_dhfs(100, 1000)
handler_positive_ion.retrieve_dhfs_results()

fig, ax = plt.subplots(1, 3)
ax[0].plot(handler_neutral_ref.rad_grid,
           (handler_neutral_ref.rv_el+handler_neutral_ref.rv_nuc)/1.44)
ax[0].plot(handler_positive_ion.rad_grid,
           (handler_positive_ion.rv_el+handler_positive_ion.rv_nuc)/1.44)
rv_test = handler_neutral_ref.rv_nuc
delta_el = handler_neutral_ref.rv_el - handler_positive_ion.rv_el
rv_test = rv_test+handler_neutral_ref.rv_el+delta_el
ax[0].plot(handler_neutral_ref.rad_grid,
           rv_test/1.44)
ax[0].set_xscale('log')

# ax[1].plot(handler_init.rad_grid,
#            handler_neutral_p1.rv_el - handler_ion_p1.rv_el)
# ax[1].plot(handler_init.rad_grid,
#            handler_neutral_p2.rv_el - handler_ion_p2.rv_el)
# ax[1].set_xscale('log')

# # construct negative ion electron potential
# delta_rv_el = handler_neutral_p1.rv_el - handler_ion_p1.rv_el
# rv_total_final = handler_final_m1.rv_nuc + handler_final_m1.rv_el - delta_rv_el
# ax[2].plot(handler_init.rad_grid, rv_total_final)
# delta_rv_el = handler_neutral_p2.rv_el - handler_ion_p2.rv_el
# rv_total_final = handler_final_m2.rv_nuc + handler_final_m2.rv_el - delta_rv_el
# ax[2].plot(handler_init.rad_grid, rv_total_final)


# ax[2].set_xscale('log')
plt.show()
