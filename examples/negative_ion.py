from xml.sax import handler
from spades.dhfs import DHFSHandler, AtomicSystem, create_ion
from spades import ph

import matplotlib.pyplot as plt

ph.verbose = 1

initial_atom = AtomicSystem(
    name="45Ca",
    electron_config="auto"
)
initial_atom.print()
neutral_atom_p1 = AtomicSystem(
    atomic_number=initial_atom.Z+1, mass_number=initial_atom.mass_number)
neutral_atom_p2 = AtomicSystem(
    atomic_number=initial_atom.Z+2, mass_number=initial_atom.mass_number)
positive_ion_p1 = create_ion(initial_atom, initial_atom.Z+1)
positive_ion_p2 = create_ion(initial_atom, initial_atom.Z+2)
negative_ion_m1 = create_ion(initial_atom, initial_atom.Z-1)
negative_ion_m2 = create_ion(initial_atom, initial_atom.Z-1)

# neutral version of final atom
final_atom_m1 = AtomicSystem(
    atomic_number=initial_atom.Z-1,
    mass_number=initial_atom.mass_number,
    electron_config="auto"
)
final_atom_m2 = AtomicSystem(
    atomic_number=initial_atom.Z-2,
    mass_number=initial_atom.mass_number,
    electron_config="auto"
)

handler_init = DHFSHandler(initial_atom, "initial_atom")
handler_init.print()
handler_init.run_dhfs(100, 1000)
handler_init.retrieve_dhfs_results()

handler_ion_p1 = DHFSHandler(positive_ion_p1, "ion_p1")
handler_ion_p1.print()
handler_ion_p1.run_dhfs(100, 1000)
handler_ion_p1.retrieve_dhfs_results()

handler_ion_p2 = DHFSHandler(positive_ion_p2, "ion_p2")
handler_ion_p2.print()
handler_ion_p2.run_dhfs(100, 1000)
handler_ion_p2.retrieve_dhfs_results()


handler_neutral_p1 = DHFSHandler(neutral_atom_p1, "neutral_p1")
handler_neutral_p1.print()
handler_neutral_p1.run_dhfs(100, 1000)
handler_neutral_p1.retrieve_dhfs_results()

handler_neutral_p2 = DHFSHandler(neutral_atom_p2, "neutral_p2")
handler_neutral_p2.print()
handler_neutral_p2.run_dhfs(100, 1000)
handler_neutral_p2.retrieve_dhfs_results()

handler_final_m1 = DHFSHandler(final_atom_m1, "final_atom_m1")
handler_final_m1.print()
handler_final_m1.run_dhfs(100, 1000)
handler_final_m1.retrieve_dhfs_results()

handler_final_m2 = DHFSHandler(final_atom_m2, "final_atom_m2")
handler_final_m2.print()
handler_final_m2.run_dhfs(100, 1000)
handler_final_m2.retrieve_dhfs_results()

fig, ax = plt.subplots(1, 3)
ax[0].plot(handler_init.rad_grid, handler_init.rv_el)
ax[0].plot(handler_ion_p1.rad_grid, handler_ion_p1.rv_el)
ax[0].plot(handler_ion_p2.rad_grid, handler_ion_p2.rv_el)
ax[0].set_xscale('log')

ax[1].plot(handler_init.rad_grid,
           handler_neutral_p1.rv_el - handler_ion_p1.rv_el)
ax[1].plot(handler_init.rad_grid,
           handler_neutral_p2.rv_el - handler_ion_p2.rv_el)
ax[1].set_xscale('log')

# construct negative ion electron potential
delta_rv_el = handler_neutral_p1.rv_el - handler_ion_p1.rv_el
rv_total_final = handler_final_m1.rv_nuc + handler_final_m1.rv_el - delta_rv_el
ax[2].plot(handler_init.rad_grid, rv_total_final)
delta_rv_el = handler_neutral_p2.rv_el - handler_ion_p2.rv_el
rv_total_final = handler_final_m2.rv_nuc + handler_final_m2.rv_el - delta_rv_el
ax[2].plot(handler_init.rad_grid, rv_total_final)


ax[2].set_xscale('log')
plt.show()
