from spades.dhfs import DHFSHandler, AtomicSystem
from spades import ph

ph.verbose = 1
initial_atom = AtomicSystem(
    name="45Ca",
    electron_config="auto"
)
initial_atom.print()

handler = DHFSHandler(initial_atom, "test")
handler.print()
handler.run_dhfs(100, 1000)
