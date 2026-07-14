"use strict";

const NeuralNetwork = createNeuralNetworkModule();
const Simulation = createSimulationModule(NeuralNetwork);
const Level = createLevelModule();
const Genetics = createGeneticsModule(NeuralNetwork);
const NEAT = createNeatModule();
const RL = createRlModule(NeuralNetwork, Simulation);
const Trainer = createTrainerModule(Simulation, NeuralNetwork, NEAT);
const Rendering = createRenderingModule(Simulation);

const DEFAULT_INPUTS = [
  "obstacleDistance", "pitDistance", "spikeDistance", "springDistance",
  "rayForward", "rayUp", "enemy1", "coin1", "velocityY"
];

const INPUT_TIPS = {
  velocityY: "current vertical speed — tells the AI whether it is rising, falling or standing still",
  onGround: "1 when standing on something, 0 while mid-air — helps time jumps",
  obstacleDistance: "distance to the next pipe ahead (1 = nothing in sight)",
  pitDistance: "distance to the next pit ahead — essential for not falling in",
  spikeDistance: "distance to the next spike patch ahead",
  springDistance: "distance to the next spring ahead — springs bounce extra high",
  enemy1: "relative x/y position of the nearest enemy ahead",
  enemy2: "relative x/y position of the second nearest enemy ahead",
  coin1: "relative x/y position of the nearest uncollected coin ahead",
  coin2: "relative x/y position of the second nearest coin ahead",
  coin3: "relative x/y position of the third nearest coin ahead",
  rayForward: "how far it can see straight ahead before hitting something solid",
  rayUp: "how far it can see straight up before hitting something solid — useful under platforms",
  tileGrid: "a width x height grid of tiles in front, each marked solid / hazard / coin — the richest (and biggest) input"
};

const OUTPUT_TIPS = [
  "do nothing this frame — cheap way to wait for an enemy to pass",
  "run left — rarely needed since the goal is to the right, but allows backing off",
  "run right — the main way forward",
  "jump straight up (only works while standing on something)",
  "jump while running right — convenience combo; off by default so the AI must learn to chain jump and right itself"
];

const BENCHMARK_SEED = 20260713;

const state = {
  algorithm: "ga",
  levelSeed: 12345,
  level: null,
  levels: [],
  mapWindow: 10,
  benchmarkLevel: null,
  inputConfig: { selected: DEFAULT_INPUTS.slice(), gridWidth: 4, gridHeight: 3 },
  actionMap: [0, 1, 2, 3], // enabled global action indices; default: all but jump+right
  sensorReader: null,
  hiddenLayers: 3,
  layerSize: 10,
  shape: null,
  populationSize: 60,
  threadCount: Math.min(4, navigator.hardwareConcurrency || 4),
  rewards: { ...Simulation.DEFAULT_REWARDS },
  neat: { speciesTarget: 8 },
  rl: { learningRate: 0.001, gamma: 0.99, epsilonDecay: 0.995, batchSize: 64, episodesPerIteration: 3 },
  training: false,
  watching: false,
  watchRun: null
};

// Each learning method keeps its own model + history, so switching preserves progress.
const algoState = {
  ga: { population: [], bestGenome: null, generation: 0, history: [], levelMarkers: [] },
  neat: { population: [], bestGenome: null, bestNetwork: null, speciesCount: 1, generation: 0, history: [], levelMarkers: [] },
  rl: { agent: null, episodeCounter: 0, generation: 0, history: [], levelMarkers: [] }
};

const ALGORITHM_INFO = {
  ga: {
    generationLabel: "gen",
    description: "evolves a fixed-size network by selection, crossover and mutation"
  },
  neat: {
    generationLabel: "gen",
    description: "neuroevolution of augmenting topologies: network structure and weights evolve together, protected by speciation"
  },
  rl: {
    generationLabel: "iter",
    description: "a single agent learns from per-step rewards with deep q-learning (experience replay + target network)"
  }
};

let trainerPool = null;
let trainToken = 0;
let watchToken = 0;

const element = id => document.getElementById(id);

const controllers = {
  ga: {
    reset() {
      const A = algoState.ga;
      A.population = Genetics.createPopulation(state.populationSize, state.shape);
      A.bestGenome = A.population[0].slice();
      A.generation = 0;
      A.history = [];
      A.levelMarkers = [];
    },
    async iterate() {
      const A = algoState.ga;
      const fitnesses = await trainerPool.evaluatePopulation(A.population);
      const result = Genetics.evolve(A.population, fitnesses, state.populationSize);
      A.population = result.population;
      A.bestGenome = result.bestGenome;
      A.generation++;
      const benchmark = Simulation.evaluate(
        result.bestGenome, state.shape, state.benchmarkLevel, state.sensorReader, state.rewards, state.actionMap
      );
      A.history.push({ best: result.bestFitness, average: result.averageFitness, benchmark });
    },
    act(sensorVector) {
      const result = NeuralNetwork.forward(algoState.ga.bestGenome, state.shape, sensorVector, true);
      const chosenIndex = NeuralNetwork.argmax(result.output);
      return { actionIndex: state.actionMap[chosenIndex], chosenIndex, outputs: result.output, activations: result.activations };
    },
    drawNetwork(actResult) {
      Rendering.drawNetwork(state.shape, algoState.ga.bestGenome,
        actResult ? actResult.activations : null, state.sensorReader.labels, outputLabels());
    },
    statsText() {
      return "";
    }
  },

  neat: {
    reset() {
      const A = algoState.neat;
      A.population = NEAT.createPopulation(state.populationSize, state.sensorReader.size, state.actionMap.length);
      A.bestGenome = NEAT.cloneGenome(A.population[0]);
      A.bestNetwork = NEAT.buildNetwork(A.bestGenome);
      A.speciesCount = 1;
      A.generation = 0;
      A.history = [];
      A.levelMarkers = [];
    },
    async iterate() {
      const A = algoState.neat;
      const fitnesses = await trainerPool.evaluatePopulation(A.population);
      const result = NEAT.evolve(A.population, fitnesses, {
        targetSize: state.populationSize,
        speciesTarget: state.neat.speciesTarget
      });
      A.population = result.population;
      A.bestGenome = result.bestGenome;
      A.bestNetwork = NEAT.buildNetwork(A.bestGenome);
      A.speciesCount = result.speciesCount;
      A.generation++;
      const network = A.bestNetwork;
      const benchmark = Simulation.evaluateWith(
        vector => state.actionMap[NeuralNetwork.argmax(network.activate(vector))],
        state.benchmarkLevel, state.sensorReader, state.rewards
      );
      A.history.push({ best: result.bestFitness, average: result.averageFitness, benchmark });
    },
    act(sensorVector) {
      const A = algoState.neat;
      const outputs = A.bestNetwork.activate(sensorVector);
      const chosenIndex = NeuralNetwork.argmax(outputs);
      return { actionIndex: state.actionMap[chosenIndex], chosenIndex, outputs };
    },
    drawNetwork(actResult) {
      const A = algoState.neat;
      Rendering.drawNeatNetwork(NEAT.toGraph(A.bestGenome),
        actResult ? A.bestNetwork.values : null, state.sensorReader.labels, outputLabels());
    },
    statsText() {
      const A = algoState.neat;
      const stats = NEAT.networkStats(A.bestGenome);
      return "species " + A.speciesCount + " · hidden " + stats.hidden;
    }
  },

  rl: {
    reset() {
      const A = algoState.rl;
      A.agent = RL.createAgent(state.shape, state.rl);
      A.episodeCounter = 0;
      A.generation = 0;
      A.history = [];
      A.levelMarkers = [];
    },
    async iterate() {
      const A = algoState.rl;
      const episodes = Math.max(1, state.rl.episodesPerIteration | 0);
      let best = -Infinity;
      let sum = 0;
      for (let e = 0; e < episodes; e++) {
        const level = state.levels[A.episodeCounter++ % state.levels.length];
        const score = RL.runEpisode(A.agent, level, state.sensorReader, state.rewards, state.actionMap);
        best = Math.max(best, score);
        sum += score;
        await new Promise(resolve => setTimeout(resolve, 0));
      }
      A.generation++;
      const agent = A.agent;
      const benchmark = Simulation.evaluateWith(
        vector => state.actionMap[RL.act(agent, vector, true)],
        state.benchmarkLevel, state.sensorReader, state.rewards
      );
      A.history.push({ best, average: sum / episodes, benchmark });
    },
    act(sensorVector) {
      const result = NeuralNetwork.forward(algoState.rl.agent.genome, state.shape, sensorVector, true, true);
      let maxAbs = 1;
      for (const value of result.output) maxAbs = Math.max(maxAbs, Math.abs(value));
      const display = Float32Array.from(result.output, value => value / maxAbs); // q-values are unbounded
      const chosenIndex = NeuralNetwork.argmax(result.output);
      return { actionIndex: state.actionMap[chosenIndex], chosenIndex, outputs: display, activations: result.activations };
    },
    drawNetwork(actResult) {
      Rendering.drawNetwork(state.shape, algoState.rl.agent.genome,
        actResult ? actResult.activations : null, state.sensorReader.labels, outputLabels());
    },
    statsText() {
      const agent = algoState.rl.agent;
      return agent ? "ε " + agent.epsilon.toFixed(2) + " · replay " + agent.buffer.size : "";
    }
  }
};

function activeController() {
  return controllers[state.algorithm];
}

function activeState() {
  return algoState[state.algorithm];
}

function readInputConfig() {
  const selected = Simulation.INPUT_DEFINITIONS
    .map(definition => definition.id)
    .filter(id => element("input_" + id).checked);
  return {
    selected: selected.length ? selected : ["obstacleDistance"],
    gridWidth: Math.max(1, Math.min(8, +element("gridWidth").value || 4)),
    gridHeight: Math.max(1, Math.min(6, +element("gridHeight").value || 3))
  };
}

function rebuildShared() {
  state.sensorReader = Simulation.createSensorReader(state.inputConfig);
  state.shape = NeuralNetwork.buildShape(
    state.sensorReader.size, state.hiddenLayers, state.layerSize, state.actionMap.length
  );
}

function outputLabels() {
  return state.actionMap.map(index => Simulation.OUTPUT_LABELS[index]);
}

function rebuildTrainerPool() {
  if (trainerPool) trainerPool.terminate();
  trainerPool = new Trainer.TrainerPool(state.threadCount);
  configureTrainerPool();
}

function configureTrainerPool() {
  trainerPool.configure(state.levels, state.shape, state.inputConfig, state.rewards, state.algorithm, state.actionMap);
}

function addMap() {
  state.levelSeed = (Math.random() * 1e9) | 0;
  state.level = Level.buildLevel(state.levelSeed);
  state.levels.push(state.level);
  while (state.levels.length > state.mapWindow) state.levels.shift();
}

function setMode(mode) {
  element("modeValue").textContent = mode;
}

function updateStats() {
  const A = activeState();
  element("genLabel").textContent = ALGORITHM_INFO[state.algorithm].generationLabel;
  element("generationValue").textContent = A.generation;
  element("bestValue").textContent = A.history.length
    ? A.history[A.history.length - 1].best | 0 : 0;
  element("algoStats").textContent = activeController().statsText();
}

function updateNetworkInfo() {
  element("inputCount").textContent = state.sensorReader.size;
  element("outputCount").textContent = state.actionMap.length;
  if (state.algorithm === "neat") {
    element("weightLabel").textContent = "connections";
    element("weightCount").textContent = NEAT.networkStats(algoState.neat.bestGenome).connections;
  } else {
    element("weightLabel").textContent = "weights";
    element("weightCount").textContent = NeuralNetwork.weightCount(state.shape);
  }
}

function stopAll() {
  state.training = false;
  state.watching = false;
  trainToken++;
  watchToken++;
  element("buttonTrain").textContent = "TRAIN";
  setMode("idle");
}

function redrawAll() {
  updateStats();
  updateNetworkInfo();
  const A = activeState();
  Rendering.drawChart(A.history, A.levelMarkers);
  activeController().drawNetwork(null);
  Rendering.drawGame(Simulation.newRun(state.level), state.level, null, null);
  layoutHelpPanel(); // panel heights may have changed
}

async function runIteration() {
  addMap();
  configureTrainerPool();
  await activeController().iterate();
}

async function trainLoop() {
  const token = ++trainToken;
  while (state.training && token === trainToken) {
    await runIteration();
    if (token !== trainToken) return;
    updateStats();
    updateNetworkInfo();
    const A = activeState();
    Rendering.drawChart(A.history, A.levelMarkers);
    activeController().drawNetwork(null);
    await new Promise(resolve => setTimeout(resolve, 0));
  }
}

function watchLoop(token) {
  if (!state.watching || token !== watchToken) return;
  if (!state.watchRun || state.watchRun.dead) {
    if (state.watchRun && state.watchRun.dead) {
      state.watching = false;
      setMode("idle");
      return;
    }
    state.watchRun = Simulation.newRun(state.level);
  }
  const controller = activeController();
  const sensorVector = state.sensorReader.read(state.watchRun, state.level);
  const actResult = controller.act(sensorVector);
  Simulation.step(state.watchRun, state.level, actResult.actionIndex);
  const debug = {
    chosenIndex: actResult.chosenIndex,
    actionLabel: Simulation.OUTPUT_LABELS[actResult.actionIndex],
    outputs: actResult.outputs,
    labels: outputLabels(),
    fitness: Simulation.fitness(state.watchRun, state.level, state.rewards)
  };
  Rendering.drawGame(state.watchRun, state.level, state.sensorReader, sensorVector, debug);
  controller.drawNetwork(actResult);
  requestAnimationFrame(() => watchLoop(token));
}

function applyAlgorithmUi() {
  for (const key of ["ga", "neat", "rl"]) {
    element("algo_" + key).classList.toggle("active", key === state.algorithm);
  }
  document.querySelectorAll("[data-algo]").forEach(node => {
    node.classList.toggle("algo-hidden", !node.getAttribute("data-algo").split(" ").includes(state.algorithm));
  });
  element("algoDescription").textContent = ALGORITHM_INFO[state.algorithm].description;
  document.querySelectorAll("[data-help-algo]").forEach(node => {
    node.classList.toggle("help-active", node.getAttribute("data-help-algo") === state.algorithm);
  });
  layoutHelpPanel(); // panel heights change when per-method rows show/hide
}

function setHelpVisible(visible) {
  element("helpPanel").classList.toggle("help-hidden", !visible);
  element("helpOpen").classList.toggle("help-hidden", visible);
  if (visible) layoutHelpPanel();
}

// Docked mode: the dialog sits right of the content as one bordered block,
// each section top-aligned with (and never taller than) the panel it explains.
function layoutHelpPanel() {
  const help = element("helpPanel");
  if (help.classList.contains("help-hidden")) return;
  const firstPanel = element("panelMethod");
  const lastPanel = element("panelProgress");
  const contentRight = firstPanel.getBoundingClientRect().right + window.scrollX;
  const available = document.documentElement.clientWidth - contentRight - 8 - 16;
  const docked = available >= 220;
  help.classList.toggle("help-docked", docked);
  const sections = help.querySelectorAll(".help-section");
  if (!docked) {
    help.style.left = help.style.top = help.style.height = help.style.width = "";
    sections.forEach(section => {
      section.style.top = "";
      section.style.maxHeight = "";
    });
    return;
  }
  const top = document.querySelector("h1").getBoundingClientRect().top + window.scrollY;
  const firstTop = firstPanel.getBoundingClientRect().top + window.scrollY;
  const bottom = lastPanel.getBoundingClientRect().bottom + window.scrollY;
  help.style.left = (contentRight + 8) + "px";
  help.style.top = top + "px";
  help.style.width = Math.min(320, available) + "px";
  help.style.height = (bottom - top) + "px";
  for (const section of sections) {
    const targetId = section.getAttribute("data-help-for");
    if (!targetId) { // intro block: fills the space beside the page header
      section.style.top = "4px";
      section.style.maxHeight = (firstTop - top - 2) + "px";
      continue;
    }
    const rect = element(targetId).getBoundingClientRect();
    section.style.top = (rect.top + window.scrollY - top + 4) + "px";
    section.style.maxHeight = (rect.height - 8) + "px";
  }
}

function setAlgorithm(id) {
  if (state.algorithm === id) return;
  stopAll();
  state.algorithm = id;
  state.watchRun = null;
  applyAlgorithmUi();
  configureTrainerPool();
  redrawAll();
}

function resetAllAlgorithms() {
  rebuildShared();
  controllers.ga.reset();
  controllers.neat.reset();
  controllers.rl.reset();
}

function onInputConfigChanged() {
  stopAll();
  state.inputConfig = readInputConfig();
  resetAllAlgorithms();
  configureTrainerPool();
  redrawAll();
}

function onShapeChanged() {
  stopAll();
  rebuildShared();
  controllers.ga.reset();
  controllers.rl.reset();
  configureTrainerPool();
  redrawAll();
}

function readOutputConfig() {
  const selected = Simulation.OUTPUT_LABELS
    .map((_, index) => index)
    .filter(index => element("output_" + index).checked);
  return selected.length ? selected : [2]; // never empty: fall back to "right"
}

function onOutputConfigChanged() {
  stopAll();
  state.actionMap = readOutputConfig();
  resetAllAlgorithms();
  configureTrainerPool();
  redrawAll();
}

function buildOutputCheckboxes() {
  const container = element("outputList");
  Simulation.OUTPUT_LABELS.forEach((outputLabel, index) => {
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id = "output_" + index;
    checkbox.checked = state.actionMap.includes(index);
    checkbox.onchange = onOutputConfigChanged;
    label.appendChild(checkbox);
    label.appendChild(document.createTextNode(outputLabel));
    const tip = document.createElement("sup");
    tip.className = "tip";
    tip.setAttribute("data-tip", OUTPUT_TIPS[index]);
    tip.textContent = "?";
    label.appendChild(tip);
    container.appendChild(label);
  });
}

function buildInputCheckboxes() {
  const container = element("inputList");
  for (const definition of Simulation.INPUT_DEFINITIONS) {
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id = "input_" + definition.id;
    checkbox.checked = DEFAULT_INPUTS.includes(definition.id);
    checkbox.onchange = onInputConfigChanged;
    label.appendChild(checkbox);
    label.appendChild(document.createTextNode(definition.label));
    if (INPUT_TIPS[definition.id]) {
      const tip = document.createElement("sup");
      tip.className = "tip";
      tip.setAttribute("data-tip", INPUT_TIPS[definition.id]);
      tip.textContent = "?";
      label.appendChild(tip);
    }
    container.appendChild(label);
  }
}

function bindControls() {
  element("helpOpen").onclick = () => setHelpVisible(true);
  element("helpClose").onclick = () => setHelpVisible(false);

  element("algo_ga").onclick = () => setAlgorithm("ga");
  element("algo_neat").onclick = () => setAlgorithm("neat");
  element("algo_rl").onclick = () => setAlgorithm("rl");

  element("gridWidth").onchange = onInputConfigChanged;
  element("gridHeight").onchange = onInputConfigChanged;

  element("hiddenLayers").oninput = event => {
    state.hiddenLayers = +event.target.value;
    element("hiddenLayersValue").textContent = state.hiddenLayers;
    onShapeChanged();
  };
  element("layerSize").oninput = event => {
    state.layerSize = +event.target.value;
    element("layerSizeValue").textContent = state.layerSize;
    onShapeChanged();
  };
  element("populationSize").oninput = event => {
    state.populationSize = +event.target.value;
    element("populationSizeValue").textContent = state.populationSize;
  };
  element("threadCount").oninput = event => {
    state.threadCount = +event.target.value;
    element("threadCountValue").textContent = state.threadCount;
    rebuildTrainerPool();
  };

  const bindNumber = (id, target, key, min, max) => {
    element(id).onchange = event => {
      const value = +event.target.value;
      if (Number.isFinite(value)) target[key] = Math.min(max, Math.max(min, value));
      event.target.value = target[key];
    };
  };
  bindNumber("speciesTarget", state.neat, "speciesTarget", 1, 20);
  bindNumber("rlLearningRate", state.rl, "learningRate", 0.00001, 0.1);
  bindNumber("rlGamma", state.rl, "gamma", 0.5, 0.999);
  bindNumber("rlEpsilonDecay", state.rl, "epsilonDecay", 0.8, 1);
  bindNumber("rlBatchSize", state.rl, "batchSize", 8, 256);
  bindNumber("rlEpisodes", state.rl, "episodesPerIteration", 1, 20);

  element("buttonTrain").onclick = () => {
    if (state.training) {
      stopAll();
      activeController().drawNetwork(null);
      return;
    }
    state.watching = false;
    state.training = true;
    element("buttonTrain").textContent = "STOP";
    setMode(state.algorithm === "rl"
      ? "training (dqn, main thread)"
      : "training on " + state.threadCount + " thread" + (state.threadCount > 1 ? "s" : ""));
    trainLoop();
  };

  element("buttonWatch").onclick = () => {
    stopAll();
    state.watching = true;
    state.watchRun = null;
    setMode("watching best");
    watchLoop(watchToken);
  };

  element("buttonLevel").onclick = () => {
    addMap();
    state.watchRun = null;
    configureTrainerPool();
    if (!state.training && !state.watching) redrawAll();
  };

  element("mapWindow").onchange = event => {
    state.mapWindow = Math.max(1, +event.target.value || 1);
    event.target.value = state.mapWindow;
    while (state.levels.length > state.mapWindow) state.levels.shift();
    configureTrainerPool();
  };

  const bindReward = (id, key) => {
    element(id).onchange = event => {
      const value = +event.target.value;
      if (Number.isFinite(value)) state.rewards[key] = value;
      event.target.value = state.rewards[key];
      configureTrainerPool();
    };
  };
  bindReward("rewardCoin", "coin");
  bindReward("rewardKill", "enemyKill");
  bindReward("rewardDeath", "death");
  bindReward("rewardJump", "jump");
  bindReward("rewardDistance", "distance");
  bindReward("rewardTime", "time");
}

function initialize() {
  state.level = Level.buildLevel(state.levelSeed);
  state.levels = [state.level];
  state.benchmarkLevel = Level.buildLevel(BENCHMARK_SEED);
  buildInputCheckboxes();
  buildOutputCheckboxes();
  bindControls();
  element("threadCount").value = state.threadCount;
  element("threadCountValue").textContent = state.threadCount;
  resetAllAlgorithms();
  rebuildTrainerPool();
  applyAlgorithmUi();
  redrawAll();
  layoutHelpPanel();
  window.addEventListener("resize", layoutHelpPanel);
}

initialize();
