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

  function forward(genome, shape, input, keepActivations, linearOutput) {
    let activation = input;
    let k = 0;
    const activations = keepActivations ? [input.slice()] : null;
    for (let layer = 1; layer < shape.length; layer++) {
      const isOutputLayer = layer === shape.length - 1;
      const output = new Float32Array(shape[layer]);
      for (let j = 0; j < shape[layer]; j++) {
        let sum = genome[k++];
        for (let i = 0; i < shape[layer - 1]; i++) sum += genome[k++] * activation[i];
        output[j] = linearOutput && isOutputLayer ? sum : Math.tanh(sum);
      }
      activation = output;
      if (keepActivations) activations.push(activation);
    }
    return { output: activation, activations };
  }

  function forwardTrain(genome, shape, input, linearOutput) {
    return forward(genome, shape, input, true, linearOutput).activations;
  }

  // Accumulates dLoss/dWeight into `gradients` (same layout as genome).
  // outputGradient = dLoss/d(output activation).
  function backward(genome, shape, activations, outputGradient, gradients, linearOutput) {
    let delta = Float32Array.from(outputGradient);
    if (!linearOutput) {
      const output = activations[shape.length - 1];
      for (let j = 0; j < delta.length; j++) delta[j] *= 1 - output[j] * output[j];
    }
    const offsets = [0];
    for (let layer = 1; layer < shape.length; layer++) {
      offsets.push(offsets[layer - 1] + shape[layer] * (shape[layer - 1] + 1));
    }
    for (let layer = shape.length - 1; layer >= 1; layer--) {
      const previous = activations[layer - 1];
      const previousDelta = new Float32Array(shape[layer - 1]);
      let k = offsets[layer - 1];
      for (let j = 0; j < shape[layer]; j++) {
        const d = delta[j];
        gradients[k++] += d;
        for (let i = 0; i < shape[layer - 1]; i++) {
          gradients[k] += d * previous[i];
          previousDelta[i] += genome[k] * d;
          k++;
        }
      }
      if (layer > 1) {
        for (let i = 0; i < previousDelta.length; i++) previousDelta[i] *= 1 - previous[i] * previous[i];
      }
      delta = previousDelta;
    }
  }

  function argmax(values) {
    let bestIndex = 0;
    for (let i = 1; i < values.length; i++) if (values[i] > values[bestIndex]) bestIndex = i;
    return bestIndex;
  }

  return { buildShape, weightCount, randomGenome, forward, forwardTrain, backward, argmax };
}
