import { call, fetchJson, guard, key, load } from "./common.mjs";

let params = null;
let started = false;

async function begin({ base, seed, fps }) {
  const manifest = await fetchJson(new URL("manifest.json", base));
  const watch = manifest.watch;
  const init = await load(base, watch.init);
  const step = await load(base, watch.step);
  postMessage({ type: "compiled", ms: init.ms + step.ms });
  let carry = call(init, [key(seed, 0x7fffffff)]);
  let counter = 0;
  const frame = () => {
    if (!params) return;
    const out = call(step, [...params, ...carry, key(seed ^ 0x5bd1e995, ++counter)]);
    carry = out.slice(0, watch.num_carry);
    const [obs, reward, done] = out.slice(watch.num_carry);
    postMessage({ type: "frame", obs, reward: reward[0], done: done[0] }, [obs.buffer]);
  };
  setInterval(frame, 1000 / fps);
}

onmessage = guard(async (data) => {
  if (data.type === "params") params = data.leaves;
  else if (data.type === "start" && !started) {
    started = true;
    await begin(data);
  }
});
