1. Toy “Quantum Microtubule Memory” (PyQt)
Purpose: Illustrate a single microtubule with discrete tubulin units, each in a notional quantum state. Upon “collapse,” the program randomly assigns each unit a 0 or 1 and compares to a hardcoded “memory pattern.”

Key Features:

Uses PyQt5 to create a main window (MainWindow) and a custom MicrotubuleWidget for drawing squares.
Each square is initially in a “superposition” (None), represented as gray.
Clicking “Collapse State” collapses every None into either 0 or 1 (blue or yellow squares).
Compares the final collapsed pattern to a predefined memory_pattern of 0s and 1s.
Educational Point:
Demonstrates a very simplified idea of quantum state collapse and memory checking, but does not simulate real quantum physics or microtubule biology.

2. Network of Microtubules for Memory (Hopfield-Like, PyQt)
Purpose: Provide a toy Hopfield-type simulation, where multiple microtubules (rows) each have several tubulin units (columns). Patterns are stored, and the network tries to recall them from a noisy state.


3. 2D Genetic Algorithm in PyQt (Bits Only)
Purpose: Illustrate a basic GA (Genetic Algorithm) evolving a single 2D bit pattern (10×10) toward a hidden target. No Hopfield network here—just a GA with real-time visualization in PyQt.


4. 2D GA + Hopfield Convergence in PyQt
Purpose: Combine a genetic algorithm with a Hopfield network to show how the best individual from the GA converges in Hopfield space. The GA tries to evolve solutions that, after Hopfield convergence, match a stored pattern.


Common Themes and Notes
All examples are toy or conceptual demonstrations.
Code uses PyQt5 for GUIs, NumPy for arrays, and standard Python libraries for randomization.
Real microtubule physics/quantum processes/biological details are far more complex than these simplified analogies.
