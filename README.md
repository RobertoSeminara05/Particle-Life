# Particle Interaction Simulation

This Python program simulates the dynamics of particles interacting via variable-range forces. Particles are displayed in a 2D universe using Matplotlib, and their
evolution is animated. Users can select different interaction rules and modify system parameters to explore emergent patterns.

![Particle Simulation Demo](simulation.gif)

This GIF shows a simulation using 625 particles, 10 types, with `step5` (max speed limiter, variable forces, and ranges) in a 100x100 universe.

---

## How to Use

1. **Run the script** in a Python environment with `numpy` and `matplotlib` installed.

2. **Menu Selection:**
   At runtime, you will be prompted to select a simulation step (interaction method) by entering a number 1–5:

   | Step | Description                                                                                                                                            |
   | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
   | 1    | Fixed distance ranges, varying force strengths — fastest emergence, simplest behavior                                                                  |
   | 2    | Variable distance ranges and forces — fast, prone to pulsating                                                                                         |
   | 3    | Damped motion with variable forces — slower but more surprising emergence                                                                              |
   | 4    | Max speed limiter with variable forces — medium speed convergence, most interesting results, especially with more types, very stable (no pulsating)    |
   | 5    | Max speed + wall collisions — same as Step 4, but walls cause rearrangement of emergent structures at each collision, creating highly dynamic patterns |

3. **Enter number of particles:**

   * Must be a positive integer ≤1000.
   * Default: 400.
   * Particles are generated on a grid (square number closest to your input).

4. **Enter number of particle types:**

   * Must be a positive integer.
   * Recommended: 2–10 for interesting results.
   * Default: 5.

5. The simulation will start in a Matplotlib window showing the animated particle system.

---

## Modifiable Parameters in the Code

* **Particle System Initialization (Class `System`):**

  * `population`: Number of particles.
  * `ncols`: Number of particle types (affects colors, forces).
  * `dist`: Particle initial distribution (currently only `"grid"` is implemented).
  * `extent`: Size of the 2D universe.
  * `symm`: Boolean, if `True` interactions are symmetric (force and bounds matrices are mirrored).

* **Particle Properties:**

  * Initial velocities (`vels`) are randomly generated within `[-8, 8]`.
  * Colors are randomly assigned per particle type.

* **Force Matrices:**

  * `force_matrix`: Governs force magnitude between types (can be manually adjusted).
  * `bound_matrix`: Distance ranges where interactions are active or peak (modifiable).

* **Simulation Steps:**
  Each step function implements different interaction rules, time steps (`ts`), damping, max speed, and optional wall collisions.

* **Animation Settings:**

  * `frames`: Number of steps to animate.
  * `interval`: Milliseconds per frame.
  * `blit`: `True` for efficient rendering.

---

## Notes on Emergent Behavior

* **Step 1:** Fastest, simplest emergent clusters. Ideal for quick tests.
* **Step 2:** Fast emergence but patterns may pulsate due to variable ranges.
* **Step 3:** Slower development, can lead to unexpected or surprising structures.
* **Step 4:** Stable and rich patterns, especially with more types; clusters converge smoothly without pulsating.
* **Step 5:** Adds walls; emergent clusters are rearranged upon collisions, producing highly dynamic, visually interesting patterns.

---

## Recommendations

* Use smaller numbers of particles (≤400) for faster rendering.
* Use Step 4 or 5 to explore rich, stable emergent structures.
* Experiment with `symm=True` for symmetric interactions — can produce more balanced clusters.
* For experimentation, `force_matrix` and `bound_matrix` can be manually edited for custom interaction behaviors.
