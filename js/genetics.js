"use strict";

function createGeneticsModule(NeuralNetwork) {
  const MUTATION_RATE = 0.08;
  const MUTATION_STRENGTH = 0.5;
  const RESET_RATE = 0.01;

  function createPopulation(size, shape) {
    return Array.from({ length: size }, () => NeuralNetwork.randomGenome(shape));
  }

  function evolve(population, fitnesses, targetSize) {
    const scored = population
      .map((genome, index) => ({ genome, fitness: fitnesses[index] }))
      .sort((a, b) => b.fitness - a.fitness);
    const eliteCount = Math.min(scored.length, Math.max(2, Math.round(targetSize * 0.1)));
    const next = scored.slice(0, eliteCount).map(entry => entry.genome.slice());
    while (next.length < targetSize) {
      const pick = () => scored[Math.min(scored.length - 1, (Math.random() * Math.random() * scored.length) | 0)].genome;
      const parentA = pick();
      const parentB = pick();
      const child = new Float32Array(parentA.length);
      const crossoverPoint = (Math.random() * parentA.length) | 0;
      for (let i = 0; i < child.length; i++) {
        child[i] = i < crossoverPoint ? parentA[i] : parentB[i];
        if (Math.random() < MUTATION_RATE) child[i] += (Math.random() * 2 - 1) * MUTATION_STRENGTH;
        if (Math.random() < RESET_RATE) child[i] = (Math.random() * 2 - 1) * 0.8;
      }
      next.push(child);
    }
    return {
      population: next.slice(0, targetSize),
      bestGenome: scored[0].genome.slice(),
      bestFitness: scored[0].fitness,
      averageFitness: scored.reduce((sum, entry) => sum + entry.fitness, 0) / scored.length
    };
  }

  return { createPopulation, evolve };
}
