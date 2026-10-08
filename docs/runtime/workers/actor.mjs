import { Inbox, call, fetchJson, guard, key, load, tick } from "./common.mjs";

const parameters = new Inbox();
const actors = new Inbox();

onmessage = guard(async ({ base, seed, port }) => {
  port.onmessage = ({ data }) => (data.type === "actor" ? actors : parameters).push(data);
  const manifest = await fetchJson(new URL("manifest.json", base));
  postMessage({ type: "compiling" });
  const rollout = await load(base, manifest.rollout);
  postMessage({ type: "compiled", ms: rollout.ms });

  let actor = (await actors.next()).leaves;
  const algorithm = new Array(manifest.num_algorithm);
  let version = 0;
  for (let index = 0; ; index++) {
    if (index !== 1) {
      const message = await parameters.next();
      message.indices.forEach((position, i) => (algorithm[position] = message.leaves[i]));
      version = message.version;
    }
    const started = performance.now();
    const out = call(rollout, [...algorithm, ...actor, key(seed ^ 0x9e3779b9, index)]);
    const ms = performance.now() - started;
    actor = out.slice(0, manifest.num_actor);
    const transitions = out.slice(manifest.num_actor, manifest.num_actor + manifest.num_transitions);
    const [laps, summary] = out.slice(manifest.num_actor + manifest.num_transitions);
    port.postMessage({ transitions, laps, summary, version }, transitions.map((leaf) => leaf.buffer));
    postMessage({ type: "rollout", rollout: index + 1, steps: manifest.batch_size, version, ms });
    await tick();
  }
});
