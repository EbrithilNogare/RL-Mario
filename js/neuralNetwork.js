"use strict";

function createNeuralNetworkModule() {
  function buildShape(inputCount, hiddenLayers, layerSize, outputCount) {
    const shape = [inputCount];
    for (let i = 0; i < hiddenLayers; i++) shape.push(layerSize);
    shape.push(outputCount);
    return shape;
  }

  function weightCount(shape) {
    let count = 0;
    for (let i = 1; i < shape.length; i++) count += shape[i] * (shape[i - 1] + 1);
    return count;
  }

  function randomGenome(shape) {
    const genome = new Float32Array(weightCount(shape));
    for (let i = 0; i < genome.length; i++) genome[i] = (Math.random() * 2 - 1) * 0.8;
    return genome;
  }

  function forward(genome, shape, input, keepActivations) {
    let activation = input;
    let k = 0;
    const activations = keepActivations ? [input.slice()] : null;
    for (let layer = 1; layer < shape.length; layer++) {
      const output = new Float32Array(shape[layer]);
      for (let j = 0; j < shape[layer]; j++) {
        let sum = genome[k++];
        for (let i = 0; i < shape[layer - 1]; i++) sum += genome[k++] * activation[i];
        output[j] = Math.tanh(sum);
      }
      activation = output;
      if (keepActivations) activations.push(activation);
    }
    return { output: activation, activations };
  }

  function argmax(values) {
    let bestIndex = 0;
    for (let i = 1; i < values.length; i++) if (values[i] > values[bestIndex]) bestIndex = i;
    return bestIndex;
  }

  return { buildShape, weightCount, randomGenome, forward, argmax };
}
