# tiny-mario-rl

A zero-dependency browser showcase where a small neural network learns to play a Mario-style platformer using a genetic algorithm. Everything is plain HTML, CSS, and JavaScript — no libraries, no build step. Just open `index.html` in a browser.

## What it does

Agents run through a procedurally generated level with pipes, stairs, pits, spikes, springs, platforms, coins, and three enemy types (goombas, flyers, and unstompable spinies). Each agent is a tiny feed-forward network whose weights are evolved: every generation the population is scored by fitness, the best genomes are kept, and the rest are bred with crossover and mutation. Difficulty ramps up along the level, and reaching the flag wins.

## Features

- **Selectable AI inputs** — choose what the network senses: distances to hazards, nearest enemy/coin positions, forward and upward sight rays, or a tile grid around the player. Changing inputs rebuilds the network.
- **Configurable network** — hidden layer count and neurons per layer.
- **Configurable fitness** — rewards for distance, coins, enemy kills, and punishments for death, jumping, and time spent.
- **Multithreaded training** — the population is evaluated in parallel Web Workers with a user-set thread count.
- **Generalization training** — the network is never reset on level change; switch levels manually or automatically every N generations so the same brain must learn level-independent skills.
- **Live visualization** — the network with activations, sensor rays and grid overlay in-game, action output bars, a level progress bar, and a fitness history chart with level-change markers.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | Page layout and controls |
| `style.css` | Styling |
| `js/neuralNetwork.js` | Feed-forward network (genome = flat weight array) |
| `js/simulation.js` | Game physics, sensors/input definitions, fitness |
| `js/level.js` | Seeded procedural level generation with difficulty scaling |
| `js/genetics.js` | Selection, crossover, mutation |
| `js/trainer.js` | Web Worker pool for parallel evaluation |
| `js/rendering.js` | Game, network, and chart canvases |
| `js/main.js` | State and UI wiring |

`mario-rl.html` is the original single-file proof of concept this project was rebuilt from.
