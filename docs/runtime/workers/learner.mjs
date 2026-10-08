import { Inbox, call, fetchJson, guard, key, load, pick, tick } from "./common.mjs";

const rollouts = new Inbox();

onmessage = guard(async ({ base, seed, port }) => {
  port.onmessage = ({ data }) => rollouts.push(data);
  const manifest = await fetchJson(new URL("manifest.json", base));
  postMessage({ type: "compiling" });
  const init = await load(base, manifest.init);
  const update = await load(base, manifest.update);
  postMessage({ type: "compiled", ms: init.ms + update.ms });

  const acting = manifest.rollout.kept.filter((i) => i < manifest.num_algorithm);
  const out = call(init, [key(seed, 0)]);
  let algorithm = out.slice(0, manifest.num_algorithm);
  const actor = out.slice(manifest.num_algorithm);
  const publish = (version) =>
    port.postMessage({ type: "params", version, indices: acting, leaves: pick(algorithm, acting) });
  port.postMessage({ type: "actor", leaves: actor });
  publish(0);
  postMessage({ type: "params", leaves: pick(algorithm, manifest.params) });

  let lastParams = performance.now();
  for (let index = 0; ; index++) {
    const rollout = await rollouts.next();
    const started = performance.now();
    algorithm = call(update, [...algorithm, ...rollout.transitions, rollout.laps, key(seed, 2 * index + 1)]);
    const ms = performance.now() - started;
    publish(index + 1);
    postMessage({ type: "update", update: index + 1, lag: index - rollout.version, summary: rollout.summary, ms });
    if (performance.now() - lastParams > 200) {
      postMessage({ type: "params", leaves: pick(algorithm, manifest.params) });
      lastParams = performance.now();
    }
    await tick();
  }
});
