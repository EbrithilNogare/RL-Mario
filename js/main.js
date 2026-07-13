"use strict";

const NeuralNetwork = createNeuralNetworkModule();
const Simulation = createSimulationModule(NeuralNetwork);
const Level = createLevelModule();
const Genetics = createGeneticsModule(NeuralNetwork);
const Trainer = createTrainerModule(Simulation);
const Rendering = createRenderingModule(Simulation);

const DEFAULT_INPUTS = [
  "obstacleDistance", "pitDistance", "spikeDistance", "springDistance",
  "rayForward", "rayUp", "enemy1", "coin1", "velocityY"
];

const BENCHMARK_SEED = 20260713;

const state = {
  levelSeed: 12345,
  level: null,
  levels: [],
  mapWindow: 10,
  benchmarkLevel: null,
  inputConfig: { selected: DEFAULT_INPUTS.slice(), gridWidth: 4, gridHeight: 3 },
  sensorReader: null,
  hiddenLayers: 3,
  layerSize: 10,
  shape: null,
  population: [],
  bestGenome: null,
  generation: 0,
  history: [],
  levelMarkers: [],
  populationSize: 60,
  threadCount: Math.min(4, navigator.hardwareConcurrency || 4),
  rewards: { ...Simulation.DEFAULT_REWARDS },
  training: false,
  watching: false,
  watchRun: null
};

let trainerPool = null;
let trainToken = 0;
let watchToken = 0;

const element = id => document.getElementById(id);

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

function rebuildNetwork() {
  state.sensorReader = Simulation.createSensorReader(state.inputConfig);
  state.shape = NeuralNetwork.buildShape(
    state.sensorReader.size, state.hiddenLayers, state.layerSize, Simulation.OUTPUT_COUNT
  );
  state.population = Genetics.createPopulation(state.populationSize, state.shape);
  state.bestGenome = state.population[0].slice();
  state.generation = 0;
  state.history = [];
  state.levelMarkers = [];
  element("inputCount").textContent = state.sensorReader.size;
  element("weightCount").textContent = NeuralNetwork.weightCount(state.shape);
}

function rebuildTrainerPool() {
  if (trainerPool) trainerPool.terminate();
  trainerPool = new Trainer.TrainerPool(state.threadCount);
  configureTrainerPool();
}

function configureTrainerPool() {
  trainerPool.configure(state.levels, state.shape, state.inputConfig, state.rewards);
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
  element("generationValue").textContent = state.generation;
  element("bestValue").textContent = state.history.length
    ? state.history[state.history.length - 1].best | 0 : 0;
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
  Rendering.drawChart(state.history, state.levelMarkers);
  Rendering.drawNetwork(state.shape, state.bestGenome, null, state.sensorReader.labels);
  Rendering.drawGame(Simulation.newRun(state.level), state.level, null, null);
}

async function runGeneration() {
  addMap();
  configureTrainerPool();
  const fitnesses = await trainerPool.evaluatePopulation(state.population);
  const result = Genetics.evolve(state.population, fitnesses, state.populationSize);
  state.population = result.population;
  state.bestGenome = result.bestGenome;
  state.generation++;
  const benchmark = Simulation.evaluate(
    result.bestGenome, state.shape, state.benchmarkLevel, state.sensorReader, state.rewards
  );
  state.history.push({ best: result.bestFitness, average: result.averageFitness, benchmark });
}

async function trainLoop() {
  const token = ++trainToken;
  while (state.training && token === trainToken) {
    await runGeneration();
    if (token !== trainToken) return;
    updateStats();
    Rendering.drawChart(state.history, state.levelMarkers);
    Rendering.drawNetwork(state.shape, state.bestGenome, null, state.sensorReader.labels);
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
  const sensorVector = state.sensorReader.read(state.watchRun, state.level);
  const result = NeuralNetwork.forward(state.bestGenome, state.shape, sensorVector, true);
  const actionIndex = NeuralNetwork.argmax(result.output);
  Simulation.step(state.watchRun, state.level, actionIndex);
  const debug = {
    actionIndex,
    actionLabel: Simulation.OUTPUT_LABELS[actionIndex],
    outputs: result.output,
    fitness: Simulation.fitness(state.watchRun, state.level, state.rewards)
  };
  Rendering.drawGame(state.watchRun, state.level, state.sensorReader, sensorVector, debug);
  Rendering.drawNetwork(state.shape, state.bestGenome, result.activations, state.sensorReader.labels);
  requestAnimationFrame(() => watchLoop(token));
}

function onInputConfigChanged() {
  stopAll();
  state.inputConfig = readInputConfig();
  rebuildNetwork();
  configureTrainerPool();
  redrawAll();
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
    container.appendChild(label);
  }
}

function bindControls() {
  element("gridWidth").onchange = onInputConfigChanged;
  element("gridHeight").onchange = onInputConfigChanged;

  element("hiddenLayers").oninput = event => {
    state.hiddenLayers = +event.target.value;
    element("hiddenLayersValue").textContent = state.hiddenLayers;
    stopAll();
    rebuildNetwork();
    configureTrainerPool();
    redrawAll();
  };
  element("layerSize").oninput = event => {
    state.layerSize = +event.target.value;
    element("layerSizeValue").textContent = state.layerSize;
    stopAll();
    rebuildNetwork();
    configureTrainerPool();
    redrawAll();
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

  element("buttonTrain").onclick = () => {
    if (state.training) {
      stopAll();
      Rendering.drawNetwork(state.shape, state.bestGenome, null, state.sensorReader.labels);
      return;
    }
    state.watching = false;
    state.training = true;
    element("buttonTrain").textContent = "STOP";
    setMode("training on " + state.threadCount + " thread" + (state.threadCount > 1 ? "s" : ""));
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
  bindControls();
  element("threadCount").value = state.threadCount;
  element("threadCountValue").textContent = state.threadCount;
  rebuildNetwork();
  rebuildTrainerPool();
  redrawAll();
}

initialize();
