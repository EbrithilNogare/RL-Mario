"use strict";

function createLevelModule() {
  const GROUND_Y = 200;
  const LEVEL_LENGTH = 6000;

  function mulberry32(seed) {
    return function () {
      seed |= 0;
      seed = seed + 0x6D2B79F5 | 0;
      let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  function addPipe(level, x, difficulty, random) {
    const height = 32
      + (random() < 0.3 + difficulty * 0.5 ? 16 : 0)
      + (random() < difficulty * 0.6 ? 16 : 0);
    level.obstacles.push({ x, w: 32, h: height });
    if (random() < 0.4) level.coins.push({ x: x + 16, y: GROUND_Y - height - 40 });
    return x + 32;
  }

  function addStairs(level, x, difficulty, random) {
    const stepCount = 2 + (random() * (2 + difficulty * 2) | 0);
    for (let i = 0; i < stepCount; i++) {
      level.obstacles.push({ x: x + i * 20, w: 20, h: 16 * (i + 1) });
    }
    if (random() < 0.5) level.coins.push({ x: x + stepCount * 20 + 20, y: GROUND_Y - 16 * stepCount - 40 });
    return x + stepCount * 20 + 30;
  }

  function addPit(level, x, difficulty, random) {
    const width = 48 + (random() * (2 + difficulty) | 0) * 12;
    level.pits.push({ x, w: width });
    if (random() < 0.6) {
      level.coins.push({ x: x + width / 2 - 24, y: GROUND_Y - 80 });
      level.coins.push({ x: x + width / 2, y: GROUND_Y - 96 });
      level.coins.push({ x: x + width / 2 + 24, y: GROUND_Y - 80 });
    }
    return x + width;
  }

  function addWalkers(level, x, difficulty, random) {
    const enemyCount = 1 + (random() < difficulty * 0.8 ? 1 : 0);
    for (let i = 0; i < enemyCount; i++) {
      const type = random() < difficulty * 0.45 ? "spiny" : "walker";
      level.enemies.push({ x: x + 50 + i * 90, y: GROUND_Y, range: 55 + random() * (55 + difficulty * 60), type });
    }
    return x + 70 + (enemyCount - 1) * 90;
  }

  function addFlyers(level, x, difficulty, random) {
    const enemyCount = 1 + (random() < difficulty * 0.6 ? 1 : 0);
    for (let i = 0; i < enemyCount; i++) {
      level.enemies.push({
        x: x + 50 + i * 110, y: GROUND_Y - 55 - random() * 25,
        range: 45 + random() * (45 + difficulty * 55), type: "flyer"
      });
    }
    return x + 70 + (enemyCount - 1) * 110;
  }

  function addCoinRow(level, x, difficulty, random) {
    const coinCount = 2 + (random() * 3 | 0);
    for (let i = 0; i < coinCount; i++) {
      level.coins.push({ x: x + i * 24, y: GROUND_Y - 70 - (random() < 0.5 ? 30 : 0) });
    }
    return x + coinCount * 24;
  }

  function addPlatform(level, x, difficulty, random) {
    const width = 80 + (random() * 3 | 0) * 16;
    const y = GROUND_Y - 64 - (random() < 0.4 ? 24 : 0);
    level.platforms.push({ x, y, w: width });
    const coinCount = width / 24 | 0;
    for (let i = 0; i < coinCount; i++) {
      level.coins.push({ x: x + 12 + i * 24, y: y - 24 });
    }
    if (random() < 0.25 + difficulty * 0.25 && width >= 96) {
      level.enemies.push({ x: x + width / 2, y, range: width / 2 - 14, type: "walker" });
    }
    return x + width;
  }

  function addDoublePlatform(level, x, difficulty, random) {
    const lowY = GROUND_Y - 56;
    const highY = GROUND_Y - 96;
    level.platforms.push({ x, y: lowY, w: 64 });
    level.platforms.push({ x: x + 88, y: highY, w: 64 });
    for (let i = 0; i < 2; i++) level.coins.push({ x: x + 16 + i * 28, y: lowY - 24 });
    for (let i = 0; i < 2; i++) level.coins.push({ x: x + 104 + i * 28, y: highY - 24 });
    if (random() < difficulty * 0.5) {
      level.enemies.push({ x: x + 90, y: GROUND_Y, range: 60, type: "walker" });
    }
    return x + 152;
  }

  function addSpikes(level, x, difficulty, random) {
    const width = 32
      + (random() < 0.4 + difficulty * 0.5 ? 16 : 0)
      + (random() < difficulty * 0.5 ? 8 : 0);
    level.spikes.push({ x, w: width });
    if (random() < 0.3) level.coins.push({ x: x + width / 2, y: GROUND_Y - 90 });
    return x + width;
  }

  function addSpring(level, x, difficulty, random) {
    level.springs.push({ x });
    for (let i = 0; i < 3; i++) {
      level.coins.push({ x: x - 16 + i * 20, y: GROUND_Y - 130 });
    }
    return x + 30;
  }

  function buildLevel(seed) {
    const random = mulberry32(seed);
    const level = {
      length: LEVEL_LENGTH,
      flagX: LEVEL_LENGTH - 160,
      obstacles: [], pits: [], coins: [], enemies: [],
      platforms: [], spikes: [], springs: []
    };
    let x = 420;
    while (x < level.flagX - 320) {
      const difficulty = Math.min(1, x / (LEVEL_LENGTH - 800));
      const roll = random();
      if (roll < 0.11 + difficulty * 0.06) x = addPipe(level, x, difficulty, random);
      else if (roll < 0.20 + difficulty * 0.06) x = addStairs(level, x, difficulty, random);
      else if (roll < 0.32 + difficulty * 0.04) x = addPit(level, x, difficulty, random);
      else if (roll < 0.44 + difficulty * 0.10) x = addWalkers(level, x, difficulty, random);
      else if (roll < 0.53 + difficulty * 0.14) x = addFlyers(level, x, difficulty, random);
      else if (roll < 0.62) x = addCoinRow(level, x, difficulty, random);
      else if (roll < 0.70) x = addDoublePlatform(level, x, difficulty, random);
      else if (roll < 0.80) x = addPlatform(level, x, difficulty, random);
      else if (roll < 0.90) x = addSpikes(level, x, difficulty, random);
      else x = addSpring(level, x, difficulty, random);
      x += 130 - difficulty * 60 + random() * (170 - difficulty * 70);
    }
    return level;
  }

  return { buildLevel, LEVEL_LENGTH, GROUND_Y };
}
