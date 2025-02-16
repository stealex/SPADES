from SPADES.dhfs import dhfs_handler, atomic_system
from SPADES import ph

ph.verbose = 1
initial_atom = atomic_system(
    name="45Ca",
    electron_config="auto"
)
initial_atom.print()

handler = dhfs_handler(initial_atom, "test")
handler.print()
handler.run_dhfs(100, 1000)
