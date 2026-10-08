const CATALOG = {
  podracers: [
    { id: "anakin", name: "Anakin", blurb: "Acting, environment steps and learning compile into one program. The fastest pod when the environment is written in JAX." },
    { id: "sebulba", name: "Sebulba", blurb: "An actor collects rollouts while a learner trains on the last batch and sends back fresh parameters, one update behind." },
    { id: "quadinaros", name: "Quadinaros", blurb: "Trains on recorded data and never steps the track. The environment only comes out to show the greedy policy." },
  ],
  pilots: [
    { id: "ppo", name: "PPO", blurb: "Clipped policy gradients with a learned value baseline. The dependable default." },
    { id: "pqn", name: "PQN", blurb: "Q-learning over many parallel environments, with no replay buffer and no target network." },
    { id: "dqn", name: "DQN", blurb: "Q-learning from a replay buffer with a slowly updated target network." },
    { id: "bc", name: "BC", blurb: "Behavior cloning: copies the actions in the recorded data." },
    { id: "iql", name: "IQL", blurb: "Implicit Q-learning: offline RL that never asks about actions outside the data." },
  ],
  tracks: [
    { id: "cartpole", name: "CartPole", blurb: "Push a cart left or right to keep a pole upright. Solved at a return of 475 out of 500." },
    { id: "breakout", name: "Breakout", blurb: "MinAtar Breakout on a 10 by 10 grid: bounce the ball and clear the bricks." },
    { id: "acrobot", name: "Acrobot", blurb: "Swing a two-link arm until its tip clears the bar. Every step costs -1, so faster is better." },
  ],
  runs: [
    { podracer: "anakin", pilot: "ppo", track: "cartpole", num_envs: 128, steps_per_update: 4096, solved: 475 },
    { podracer: "anakin", pilot: "pqn", track: "cartpole", num_envs: 128, steps_per_update: 4096, solved: 475 },
    { podracer: "anakin", pilot: "dqn", track: "cartpole", num_envs: 64, steps_per_update: 2048, solved: 475 },
    { podracer: "anakin", pilot: "ppo", track: "breakout", num_envs: 128, steps_per_update: 4096, solved: 30 },
    { podracer: "anakin", pilot: "pqn", track: "breakout", num_envs: 128, steps_per_update: 4096, solved: 30 },
    { podracer: "anakin", pilot: "ppo", track: "acrobot", num_envs: 64, steps_per_update: 2048, solved: -100 },
    { podracer: "anakin", pilot: "pqn", track: "acrobot", num_envs: 64, steps_per_update: 2048, solved: -100 },
    { podracer: "anakin", pilot: "dqn", track: "acrobot", num_envs: 64, steps_per_update: 2048, solved: -100 },
    { podracer: "sebulba", pilot: "ppo", track: "cartpole", num_envs: 64, steps_per_update: 2048, solved: 475 },
    { podracer: "sebulba", pilot: "pqn", track: "cartpole", num_envs: 64, steps_per_update: 2048, solved: 475 },
    { podracer: "sebulba", pilot: "ppo", track: "breakout", num_envs: 64, steps_per_update: 2048, solved: 30 },
    { podracer: "sebulba", pilot: "ppo", track: "acrobot", num_envs: 64, steps_per_update: 2048, solved: -100 },
    { podracer: "quadinaros", pilot: "bc", track: "cartpole", num_envs: 64, steps_per_update: 2048, solved: 475 },
    { podracer: "quadinaros", pilot: "iql", track: "cartpole", num_envs: 64, steps_per_update: 2048, solved: 475 },
    { podracer: "quadinaros", pilot: "bc", track: "acrobot", num_envs: 64, steps_per_update: 2048, solved: -100 },
    { podracer: "quadinaros", pilot: "bc", track: "breakout", num_envs: 64, steps_per_update: 2048, solved: 30 },
  ],
};

const SPEED = { anakin: 42000, sebulba: 13000, quadinaros: 61000 };
const TRACK_SPEED = { cartpole: 1, breakout: 0.32, acrobot: 0.8 };
const PILOT_SPEED = { ppo: 1, pqn: 1.2, dqn: 0.55, bc: 1.3, iql: 0.8 };
const PILOT_PACE = { ppo: 1, pqn: 0.8, dqn: 1.5, bc: 0.45, iql: 0.7 };
const CURVE = {
  cartpole: { floor: 18, top: 500, midpoint: 45, width: 9 },
  breakout: { floor: 0.6, top: 38, midpoint: 150, width: 40 },
  acrobot: { floor: -500, top: -75, midpoint: 40, width: 9 },
};

export async function catalog() {
  return JSON.parse(JSON.stringify(CATALOG));
}

function generator(seed) {
  let state = (seed * 2654435761 + 1) >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function cartpole(random) {
  let x = 0, speed = 0, theta = 0, spin = 0, steps = 0;
  const reset = () => {
    x = (random() - 0.5) * 0.1; speed = (random() - 0.5) * 0.1;
    theta = (random() - 0.5) * 0.1; spin = (random() - 0.5) * 0.1; steps = 0;
  };
  reset();
  const step = (skill) => {
    for (let i = 0; i < 2; i++) {
      const greedy = theta * 10 + spin * 1.6 + x * 0.25 + speed * 0.7 > 0 ? 1 : 0;
      const action = random() < 0.5 * (1 - skill) ? (random() < 0.5 ? 0 : 1) : greedy;
      const force = action ? 10 : -10;
      const cos = Math.cos(theta), sin = Math.sin(theta);
      const temp = (force + 0.05 * spin * spin * sin) / 1.1;
      const angular = (9.8 * sin - cos * temp) / (0.5 * (4 / 3 - (0.1 * cos * cos) / 1.1));
      const linear = temp - (0.05 * angular * cos) / 1.1;
      x += 0.02 * speed; speed += 0.02 * linear;
      theta += 0.02 * spin; spin += 0.02 * angular;
      steps += 1;
      if (Math.abs(x) > 2.4 || Math.abs(theta) > 0.2095 || steps >= 500) reset();
    }
    return { track: "cartpole", x, theta };
  };
  return step;
}

function breakout(random) {
  const height = 10, width = 10, channels = 4;
  let bricks, ball, last, direction, paddle, tick = 0;
  const fill = () => { bricks = new Uint8Array(height * width); for (let r = 1; r < 4; r++) for (let c = 0; c < width; c++) bricks[r * width + c] = 1; };
  const serve = () => { ball = [3, Math.floor(random() * width)]; last = ball.slice(); direction = [1, random() < 0.5 ? -1 : 1]; };
  fill(); serve(); paddle = 4;
  const landing = () => {
    let [r, c] = ball, [dr, dc] = direction;
    for (let i = 0; i < 40 && r < height - 1; i++) {
      r += dr; c += dc;
      if (c < 0) { c = 0; dc = 1; } if (c > width - 1) { c = width - 1; dc = -1; }
      if (r < 0) { r = 0; dr = 1; }
    }
    return c;
  };
  return (skill) => {
    tick += 1;
    if (tick % 3 === 0) {
      const target = random() < skill ? landing() : Math.floor(random() * width);
      paddle += Math.sign(target - paddle);
      last = ball.slice();
      let [r, c] = [ball[0] + direction[0], ball[1] + direction[1]];
      if (c < 0 || c >= width) { direction[1] *= -1; c = ball[1] + direction[1]; }
      if (r < 0) { direction[0] = 1; r = 1; }
      if (r >= 0 && r < height && bricks[r * width + c]) { bricks[r * width + c] = 0; direction[0] *= -1; r = ball[0]; }
      if (r === height - 1) {
        if (Math.abs(c - paddle) <= 0) { direction[0] = -1; r = height - 2; }
        else { serve(); r = ball[0]; c = ball[1]; }
      }
      ball = [r, c];
      if (!bricks.some(Boolean)) fill();
    }
    const grid = new Uint8Array(height * width * channels);
    const set = (r, c, k) => { if (r >= 0 && r < height && c >= 0 && c < width) grid[(r * width + c) * channels + k] = 1; };
    set(height - 1, paddle, 0);
    set(last[0], last[1], 2);
    set(ball[0], ball[1], 1);
    for (let i = 0; i < height * width; i++) if (bricks[i]) grid[i * channels + 3] = 1;
    return { track: "breakout", grid, height, width, channels };
  };
}

function catcher(random) {
  const height = 10, width = 5, channels = 2;
  let ball, paddle = 2, tick = 0;
  const drop = () => { ball = [0, Math.floor(random() * width)]; };
  drop();
  return (skill) => {
    tick += 1;
    if (tick % 4 === 0) {
      const target = random() < skill ? ball[1] : Math.floor(random() * width);
      paddle = Math.max(0, Math.min(width - 1, paddle + Math.sign(target - paddle)));
      ball[0] += 1;
      if (ball[0] >= height - 1) drop();
    }
    const grid = new Uint8Array(height * width * channels);
    grid[((height - 1) * width + paddle) * channels] = 1;
    grid[(ball[0] * width + ball[1]) * channels + 1] = 1;
    return { track: "catch", grid, height, width, channels };
  };
}

function acrobot(random) {
  let clock = 0;
  return (skill) => {
    clock += 1 / 30;
    const reach = 0.4 + 2.6 * skill;
    const theta1 = Math.sin(clock * 2.2) * reach + (random() - 0.5) * 0.05;
    const theta2 = Math.sin(clock * 2.2 + 0.9) * reach * 0.8;
    return { track: "acrobot", theta1, theta2 };
  };
}

const PLAYERS = { cartpole, breakout, acrobot, catch: catcher };

export async function start(config, handlers = {}) {
  const { podracer, pilot, track, seed = 0 } = config;
  const run = CATALOG.runs.find((entry) => entry.podracer === podracer && entry.pilot === pilot && entry.track === track);
  if (!run) throw new Error(`No run for ${podracer} / ${pilot} / ${track}`);
  const { onStatus = () => {}, onUpdate = () => {}, onFrame = () => {} } = handlers;
  const random = generator(seed + 7);
  const play = PLAYERS[track](generator(seed + 11));
  const curve = CURVE[track];
  const pace = PILOT_PACE[pilot] || 1;
  const sps = SPEED[podracer] * TRACK_SPEED[track] * (PILOT_SPEED[pilot] || 1);
  const timers = [];
  let stopped = false, update = 0, steps = 0, skill = 0;
  const recent = [];
  const later = (delay, action) => timers.push(setTimeout(() => { if (!stopped) action(); }, delay));
  const progress = (count) => 1 / (1 + Math.exp(-(count - curve.midpoint * pace) / (curve.width * pace)));

  const tick = () => {
    const jitter = 0.9 + 0.2 * random();
    const now = performance.now();
    const interval = Math.max(40, (run.steps_per_update / sps) * 1000 * jitter);
    update += 1;
    steps += run.steps_per_update;
    recent.push(now);
    if (recent.length > 6) recent.shift();
    const elapsed = recent.length > 1 ? (recent[recent.length - 1] - recent[0]) / 1000 : interval / 1000;
    const measured = recent.length > 1 ? ((recent.length - 1) * run.steps_per_update) / elapsed : sps;
    skill = progress(update);
    const noise = (random() - 0.5) * 0.18 * (curve.top - curve.floor) * (1 - skill * 0.7);
    const value = curve.floor + (curve.top - curve.floor) * skill + noise;
    const episodeReturn = update < 3 ? NaN : Math.max(Math.min(curve.floor, -1), Math.min(curve.top, value));
    const report = { update, steps, sps: measured, episodeReturn };
    if (podracer === "sebulba") report.lag = 1;
    onUpdate(report);
    later(interval, tick);
  };

  onStatus({ phase: "loading", message: "Loading the WebAssembly module" });
  const compileMs = 900 + Math.round(random() * 700);
  later(500, () => {
    onStatus({ phase: "compiling", message: podracer === "sebulba" ? "Compiling the actor and the learner" : "Compiling the train step" });
    later(compileMs, () => {
      onStatus({ phase: "training", message: "Training", compileMs });
      later(10, tick);
      const frame = () => { onFrame(play(skill)); };
      timers.push(setInterval(() => { if (!stopped) frame(); }, 33));
    });
  });

  return {
    stop() {
      if (stopped) return;
      stopped = true;
      timers.forEach((timer) => { clearTimeout(timer); clearInterval(timer); });
      onStatus({ phase: "stopped", message: "Stopped" });
    },
  };
}
