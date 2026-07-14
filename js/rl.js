"use strict";

// Deep Q-learning, tuned for this game:
// - action repeat: one decision per 4 physics frames — 4x shorter credit-assignment
//   horizon and 4x cheaper episodes
// - n-step (3) returns — rewards propagate several states per update
// - Double DQN — online net picks the next action, target net values it
// - ReLU hidden layers + He init — trains far better than tanh here
// - soft (Polyak) target updates — smoother than periodic hard syncs
// - timeout/stuck deaths bootstrap (only real deaths and wins are terminal)
// Per-decision reward is the fitness delta, so an episode's return telescopes to
// exactly the fitness GA/NEAT optimize.
function createRlModule(NeuralNetwork, Simulation) {
  const REWARD_SCALE = 100;
  const REPLAY_CAPACITY = 20000;
  const WARMUP_TRANSITIONS = 300;
  const ACTION_REPEAT = 4;
  const N_STEP = 3;
  const TARGET_TAU = 0.01;
  const UPDATES_PER_DECISION = 2;
  const EPSILON_MIN = 0.05;
  const ERROR_CLIP = 1;
  const ADAM_BETA1 = 0.9;
  const ADAM_BETA2 = 0.999;
  const ADAM_EPSILON = 1e-8;

  function createAgent(shape, config) {
    const genome = NeuralNetwork.heGenome(shape);
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

  function qValues(genome, shape, input) {
    return NeuralNetwork.forward(genome, shape, input, false, true, true).output;
  }

  function act(agent, input, greedy) {
    const outputCount = agent.shape[agent.shape.length - 1];
    if (!greedy && Math.random() < agent.epsilon) return (Math.random() * outputCount) | 0;
    return NeuralNetwork.argmax(qValues(agent.genome, agent.shape, input));
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
    const bootstrapGamma = Math.pow(config.gamma, N_STEP); // stored rewards span N_STEP decisions
    agent.gradients.fill(0);
    for (let n = 0; n < batchSize; n++) {
      const i = (Math.random() * buffer.size) | 0;
      const activations = NeuralNetwork.forwardTrain(agent.genome, shape, buffer.states[i], true, true);
      const q = activations[activations.length - 1];
      let targetValue = buffer.rewards[i];
      if (!buffer.dones[i]) {
        // Double DQN: online net chooses, target net evaluates.
        const onlineNext = qValues(agent.genome, shape, buffer.nextStates[i]);
        const targetNext = qValues(agent.target, shape, buffer.nextStates[i]);
        targetValue += bootstrapGamma * targetNext[NeuralNetwork.argmax(onlineNext)];
      }
      const action = buffer.actions[i];
      const error = Math.max(-ERROR_CLIP, Math.min(ERROR_CLIP, q[action] - targetValue));
      const outputGradient = new Float32Array(outputCount);
      outputGradient[action] = error / batchSize;
      NeuralNetwork.backward(agent.genome, shape, activations, outputGradient, agent.gradients, true, true);
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
      agent.target[i] += TARGET_TAU * (agent.genome[i] - agent.target[i]);
    }
    agent.updates++;
  }

  // actionMap: network output index -> global action index.
  // The replay buffer stores network indices; only step() sees global actions.
  function runEpisode(agent, level, sensorReader, rewards, actionMap) {
    const config = agent.config;
    const run = Simulation.newRun(level);
    const pending = []; // last N_STEP transitions awaiting their n-step return
    const flush = (nextState, done) => {
      let nStepReward = 0;
      for (let i = pending.length - 1; i >= 0; i--) nStepReward = pending[i].reward + config.gamma * nStepReward;
      const first = pending.shift();
      remember(agent, first.state, first.action, nStepReward, nextState, done);
    };
    let sensorVector = sensorReader.read(run, level);
    while (!run.dead) {
      const action = act(agent, sensorVector, false);
      const fitnessBefore = Simulation.fitness(run, level, rewards);
      for (let k = 0; k < ACTION_REPEAT && !run.dead; k++) Simulation.step(run, level, actionMap[action]);
      const fitnessAfter = Simulation.fitness(run, level, rewards);
      const nextVector = sensorReader.read(run, level);
      pending.push({ state: sensorVector, action, reward: (fitnessAfter - fitnessBefore) / REWARD_SCALE });
      // timeout/stuck is a time limit, not a real outcome — keep bootstrapping there
      const terminal = run.killedByHazard || run.won;
      if (run.dead) {
        while (pending.length) flush(nextVector, terminal);
      } else if (pending.length === N_STEP) {
        flush(nextVector, false);
      }
      sensorVector = nextVector;
      agent.stepCount++;
      if (agent.buffer.size >= WARMUP_TRANSITIONS) {
        for (let u = 0; u < UPDATES_PER_DECISION; u++) trainBatch(agent);
      }
    }
    agent.epsilon = Math.max(EPSILON_MIN, agent.epsilon * config.epsilonDecay);
    return Simulation.fitness(run, level, rewards);
  }

  return { createAgent, act, runEpisode };
}
