"use strict";

function createTrainerModule(Simulation) {
  function buildWorkerSource() {
    return [
      '"use strict";',
      `const NeuralNetwork = (${createNeuralNetworkModule.toString()})();`,
      `const Simulation = (${createSimulationModule.toString()})(NeuralNetwork);`,
      "let levels = [], shape = null, sensorReader = null, rewards = null;",
      "onmessage = (event) => {",
      "  const message = event.data;",
      "  if (message.type === 'configure') {",
      "    levels = message.levels;",
      "    shape = message.shape;",
      "    sensorReader = Simulation.createSensorReader(message.inputConfig);",
      "    rewards = message.rewards;",
      "    return;",
      "  }",
      "  const fitnesses = new Float64Array(message.genomes.length);",
      "  for (let i = 0; i < message.genomes.length; i++) {",
      "    let sum = 0;",
      "    for (let m = 0; m < levels.length; m++) sum += Simulation.evaluate(message.genomes[i], shape, levels[m], sensorReader, rewards);",
      "    fitnesses[i] = levels.length ? sum / levels.length : 0;",
      "  }",
      "  postMessage({ jobId: message.jobId, fitnesses }, [fitnesses.buffer]);",
      "};"
    ].join("\n");
  }

  class TrainerPool {
    constructor(threadCount) {
      this.workers = [];
      this.nextJobId = 1;
      this.pendingJobs = new Map();
      this.levels = [];
      this.shape = null;
      this.sensorReader = null;
      this.rewards = null;
      try {
        const url = URL.createObjectURL(new Blob([buildWorkerSource()], { type: "text/javascript" }));
        for (let i = 0; i < threadCount; i++) {
          const worker = new Worker(url);
          worker.onmessage = (event) => this.handleMessage(event.data);
          worker.onerror = () => this.disableWorkers();
          this.workers.push(worker);
        }
        URL.revokeObjectURL(url);
      } catch (error) {
        this.disableWorkers();
      }
    }

    handleMessage(data) {
      const job = this.pendingJobs.get(data.jobId);
      if (!job) return;
      this.pendingJobs.delete(data.jobId);
      job.resolve(data.fitnesses);
    }

    disableWorkers() {
      for (const worker of this.workers) worker.terminate();
      this.workers = [];
      for (const [jobId, job] of this.pendingJobs) {
        this.pendingJobs.delete(jobId);
        job.resolve(this.evaluateSync(job.genomes));
      }
    }

    configure(levels, shape, inputConfig, rewards) {
      this.levels = levels;
      this.shape = shape;
      this.sensorReader = Simulation.createSensorReader(inputConfig);
      this.rewards = rewards;
      for (const worker of this.workers) {
        worker.postMessage({ type: "configure", levels, shape, inputConfig, rewards });
      }
    }

    evaluateSync(genomes) {
      const fitnesses = new Float64Array(genomes.length);
      for (let i = 0; i < genomes.length; i++) {
        let sum = 0;
        for (const level of this.levels) {
          sum += Simulation.evaluate(genomes[i], this.shape, level, this.sensorReader, this.rewards);
        }
        fitnesses[i] = this.levels.length ? sum / this.levels.length : 0;
      }
      return fitnesses;
    }

    async evaluatePopulation(population) {
      if (!this.workers.length) return this.evaluateSync(population);
      const fitnesses = new Float64Array(population.length);
      const chunkCount = Math.min(this.workers.length, population.length);
      const chunkSize = Math.ceil(population.length / chunkCount);
      const promises = [];
      for (let chunkIndex = 0; chunkIndex < chunkCount; chunkIndex++) {
        const start = chunkIndex * chunkSize;
        const genomes = population.slice(start, start + chunkSize);
        if (!genomes.length) continue;
        promises.push(new Promise(resolve => {
          const jobId = this.nextJobId++;
          this.pendingJobs.set(jobId, { resolve, genomes });
          this.workers[chunkIndex].postMessage({ type: "evaluate", jobId, genomes });
        }).then(chunkFitnesses => fitnesses.set(chunkFitnesses, start)));
      }
      await Promise.all(promises);
      return fitnesses;
    }

    terminate() {
      for (const worker of this.workers) worker.terminate();
      this.workers = [];
    }
  }

  return { TrainerPool };
}
