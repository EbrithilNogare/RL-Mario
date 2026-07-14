"use strict";

// Deep Q-learning (DQN): experience replay + target network, trained by backprop.
// The Q-network uses the same flat-genome MLP format as the GA, so the network
// visualization and watch loop work unchanged. Per-step reward is the fitness
// delta, so an episode's return telescopes to exactly the fitness GA/NEAT optimize.
function createRlModule(NeuralNetwork, Simulation) {
  const REWARD_SCALE = 100;
  const REPLAY_CAPACITY = 20000;
  const WARMUP_TRANSITIONS = 500;
  const TRAIN_EVERY_STEPS = 4;
  const TARGET_SYNC_UPDATES = 200;
  const EPSILON_MIN = 0.05;
  const ADAM_BETA1 = 0.9;
  const ADAM_BETA2 = 0.999;
  const ADAM_EPSILON = 1e-8;

  function createAgent(shape, config) {
    const genome = NeuralNetwork.randomGenome(shape);
    for (let i = 0; i < genome.length; i++) genome[i] *= 0.4; // smaller init for gradient training
    return {
      shape, config, genome,
      target: genome.slice(),
      gradients: new Float32Array(genome.length),
      adamM: new Float32Array(genome.length),
      adamV: new Float32Array(genome.length),
      adamT: 0,
      epsilon: 1,
      updates: 0,
      stepCount: 0,
      buffer: { states: [], actions: [], rewards: [], nextStates: [], dones: [], index: 0, size: 0 }
    };
  }

  function act(agent, input, greedy) {
    const outputCount = agent.shape[agent.shape.length - 1];
    if (!greedy && Math.random() < agent.epsilon) return (Math.random() * outputCount) | 0;
    const q = NeuralNetwork.forward(agent.genome, agent.shape, input, false, true).output;
    return NeuralNetwork.argmax(q);
  }

  function remember(agent, state, action, reward, nextState, done) {
    const buffer = agent.buffer;
    if (buffer.size < REPLAY_CAPACITY) {
      buffer.states.push(state);
      buffer.actions.push(action);
      buffer.rewards.push(reward);
      buffer.nextStates.push(nextState);
      buffer.dones.push(done);
      buffer.size++;
    } else {
      const i = buffer.index;
      buffer.states[i] = state;
      buffer.actions[i] = action;
      buffer.rewards[i] = reward;
      buffer.nextStates[i] = nextState;
      buffer.dones[i] = done;
      buffer.index = (i + 1) % REPLAY_CAPACITY;
    }
  }

  function trainBatch(agent) {
    const { config, buffer, shape } = agent;
    const batchSize = Math.min(Math.max(1, config.batchSize | 0), buffer.size);
    const outputCount = shape[shape.length - 1];
    agent.gradients.fill(0);
    for (let n = 0; n < batchSize; n++) {
      const i = (Math.random() * buffer.size) | 0;
      const activations = NeuralNetwork.forwardTrain(agent.genome, shape, buffer.states[i], true);
      const q = activations[activations.length - 1];
      let targetValue = buffer.rewards[i];
      if (!buffer.dones[i]) {
        const nextQ = NeuralNetwork.forward(agent.target, shape, buffer.nextStates[i], false, true).output;
        let maxQ = nextQ[0];
        for (let j = 1; j < nextQ.length; j++) maxQ = Math.max(maxQ, nextQ[j]);
        targetValue += config.gamma * maxQ;
      }
      const action = buffer.actions[i];
      const error = Math.max(-1, Math.min(1, q[action] - targetValue)); // clipped TD error
      const outputGradient = new Float32Array(outputCount);
      outputGradient[action] = error / batchSize;
      NeuralNetwork.backward(agent.genome, shape, activations, outputGradient, agent.gradients, true);
    }
    agent.adamT++;
    const correction1 = 1 - Math.pow(ADAM_BETA1, agent.adamT);
    const correction2 = 1 - Math.pow(ADAM_BETA2, agent.adamT);
    for (let i = 0; i < agent.genome.length; i++) {
      const gradient = agent.gradients[i];
      agent.adamM[i] = ADAM_BETA1 * agent.adamM[i] + (1 - ADAM_BETA1) * gradient;
      agent.adamV[i] = ADAM_BETA2 * agent.adamV[i] + (1 - ADAM_BETA2) * gradient * gradient;
      agent.genome[i] -= config.learningRate * (agent.adamM[i] / correction1)
        / (Math.sqrt(agent.adamV[i] / correction2) + ADAM_EPSILON);
    }
    agent.updates++;
    if (agent.updates % TARGET_SYNC_UPDATES === 0) agent.target.set(agent.genome);
  }

  // actionMap: network output index -> global action index.
  // The replay buffer stores network indices; only step() sees global actions.
  function runEpisode(agent, level, sensorReader, rewards, actionMap) {
    const run = Simulation.newRun(level);
    let sensorVector = sensorReader.read(run, level);
    while (!run.dead) {
      const action = act(agent, sensorVector, false);
      const fitnessBefore = Simulation.fitness(run, level, rewards);
      Simulation.step(run, level, actionMap[action]);
      const fitnessAfter = Simulation.fitness(run, level, rewards);
      const nextVector = sensorReader.read(run, level);
      remember(agent, sensorVector, action, (fitnessAfter - fitnessBefore) / REWARD_SCALE, nextVector, run.dead);
      sensorVector = nextVector;
      agent.stepCount++;
      if (agent.buffer.size >= WARMUP_TRANSITIONS && agent.stepCount % TRAIN_EVERY_STEPS === 0) {
        trainBatch(agent);
      }
    }
    agent.epsilon = Math.max(EPSILON_MIN, agent.epsilon * agent.config.epsilonDecay);
    return Simulation.fitness(run, level, rewards);
  }

  return { createAgent, act, runEpisode };
}
