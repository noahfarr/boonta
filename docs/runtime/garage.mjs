const HERE = new URL("./", import.meta.url);

let menu = null;

export async function catalog() {
  if (!menu) {
    const response = await fetch(new URL("catalog.json", HERE));
    if (!response.ok) throw new Error(`catalog.json: HTTP ${response.status}`);
    menu = await response.json();
  }
  return menu;
}

const FRAMES = {
  cartpole: (track, obs) => ({ track, x: obs[0], theta: obs[2] }),
  acrobot: (track, obs) => ({
    track,
    theta1: Math.atan2(obs[1], obs[0]),
    theta2: Math.atan2(obs[3], obs[2]),
  }),
  grid: (track, obs, shape) => {
    const [height, width, channels = 1] = shape;
    const grid = new Uint8Array(obs.length);
    for (let i = 0; i < obs.length; i++) grid[i] = obs[i] > 0.5 ? 1 : 0;
    return { track, grid, height, width, channels };
  },
};

function spawn(name) {
  return new Worker(new URL(`workers/${name}.mjs`, HERE), { type: "module" });
}

class Speedometer {
  constructor(window = 3000) {
    this.window = window;
    this.samples = [];
    this.value = 0;
  }
  record(steps) {
    const now = performance.now();
    this.samples.push([now, steps]);
    while (this.samples.length > 2 && now - this.samples[1][0] > this.window) this.samples.shift();
    const [t0, s0] = this.samples[0];
    if (now - t0 >= 250) this.value = ((steps - s0) * 1000) / (now - t0);
    return this.value;
  }
}

export async function start(selection, handlers = {}) {
  const { podracer, pilot, track, seed = 0 } = selection;
  const onStatus = handlers.onStatus ?? (() => {});
  const onUpdate = handlers.onUpdate ?? (() => {});
  const onFrame = handlers.onFrame ?? (() => {});

  onStatus({ phase: "loading", message: `loading ${podracer} / ${pilot} / ${track}` });
  const { runs, tracks } = await catalog();
  const run = runs.find((r) => r.podracer === podracer && r.pilot === pilot && r.track === track);
  if (!run) throw new Error(`no run for ${podracer} / ${pilot} / ${track}`);
  const trackInfo = tracks.find((t) => t.id === track);
  const base = new URL(run.path, HERE).href;
  const manifest = await (await fetch(new URL("manifest.json", base))).json();
  const shape = manifest.watch.obs_shape;
  const toFrame = FRAMES[trackInfo.frame];

  const workers = [];
  let stopped = false;
  const speed = new Speedometer();
  let pending = 0;
  let compileMs = 0;
  let compileStarted = 0;
  let training = false;

  const stop = () => {
    if (stopped) return;
    stopped = true;
    for (const worker of workers) worker.terminate();
    onStatus({ phase: "stopped", message: "stopped" });
  };
  const fail = (message) => {
    if (stopped) return;
    stopped = true;
    for (const worker of workers) worker.terminate();
    onStatus({ phase: "error", message });
  };
  const watcher = spawn("watch");
  workers.push(watcher);
  const forward = (message) => {
    if (message.type === "params") watcher.postMessage(message);
  };
  const attend = (worker, handle) => {
    workers.push(worker);
    pending++;
    worker.onerror = (event) => fail(event.message ?? String(event));
    worker.onmessage = ({ data }) => {
      if (stopped) return;
      if (data.type === "error") return fail(data.message);
      if (data.type === "compiling") {
        if (!compileStarted) {
          compileStarted = performance.now();
          onStatus({ phase: "compiling", message: "compiling StableHLO to WebAssembly" });
        }
        return;
      }
      if (data.type === "compiled") {
        compileMs = Math.max(compileMs, performance.now() - compileStarted);
        if (--pending === 0) {
          training = true;
          onStatus({ phase: "training", message: "training", compileMs });
        }
        return;
      }
      if (data.type === "params") return forward(data);
      handle(data);
    };
  };

  watcher.onmessage = ({ data }) => {
    if (stopped) return;
    if (data.type === "error") return fail(data.message);
    if (data.type === "frame") onFrame({ ...toFrame(track, data.obs, shape), reward: data.reward, done: data.done });
  };
  watcher.onerror = (event) => fail(event.message ?? String(event));
  watcher.postMessage({ type: "start", base, seed, fps: trackInfo.fps ?? 30 });

  let update = 0;
  let steps = 0;
  const emit = (fields) => onUpdate({ update, steps, sps: speed.record(steps), ...fields });

  if (podracer === "anakin" || podracer === "quadinaros") {
    const worker = spawn(podracer);
    attend(worker, (data) => {
      if (data.type !== "updates") return;
      if (data.updates === 0 && data.summary && data.summary[1] > 0) {
        emit({ episodeReturn: data.summary[0] / data.summary[1] });
        return;
      }
      for (let i = 0; i < data.updates; i++) {
        update++;
        steps += data.batchSize;
        let episodeReturn = NaN;
        if (data.episodes) {
          const total = data.episodes[2 * i];
          const count = data.episodes[2 * i + 1];
          if (count > 0) episodeReturn = total / count;
        } else if (data.summary && i === data.updates - 1 && data.summary[1] > 0) {
          episodeReturn = data.summary[0] / data.summary[1];
        }
        emit({ episodeReturn });
      }
    });
    worker.postMessage({ base, seed });
  } else if (podracer === "sebulba") {
    const channel = new MessageChannel();
    const learner = spawn("learner");
    const actor = spawn("actor");
    attend(actor, (data) => {
      if (data.type === "rollout") steps += data.steps;
    });
    attend(learner, (data) => {
      if (data.type !== "update") return;
      update = data.update;
      const [total, count] = data.summary;
      emit({ episodeReturn: count > 0 ? total / count : NaN, lag: data.lag });
    });
    learner.postMessage({ base, seed, port: channel.port1 }, [channel.port1]);
    actor.postMessage({ base, seed, port: channel.port2 }, [channel.port2]);
  } else {
    throw new Error(`unknown podracer ${podracer}`);
  }

  return {
    stop,
    get training() {
      return training;
    },
  };
}
