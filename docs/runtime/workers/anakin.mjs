import { call, fetchJson, guard, key, load, pick, tick } from "./common.mjs";

onmessage = guard(async ({ base, seed }) => {
  const manifest = await fetchJson(new URL("manifest.json", base));
  postMessage({ type: "compiling" });
  const init = await load(base, manifest.init);
  const train = await load(base, manifest.train);
  postMessage({ type: "compiled", ms: init.ms + train.ms });

  let state = call(init, [key(seed, 0)]);
  postMessage({ type: "params", leaves: pick(state, manifest.params) });
  const updates = manifest.updates_per_call;
  let lastParams = performance.now();
  for (let counter = 1; ; counter++) {
    const started = performance.now();
    const out = call(train, [...state, key(seed, counter)]);
    const ms = performance.now() - started;
    state = out.slice(0, manifest.num_state);
    const episodes = out[manifest.num_state];
    postMessage({ type: "updates", updates, batchSize: manifest.batch_size, episodes, ms }, [episodes.buffer]);
    if (performance.now() - lastParams > 200) {
      postMessage({ type: "params", leaves: pick(state, manifest.params) });
      lastParams = performance.now();
    }
    await tick();
  }
});
