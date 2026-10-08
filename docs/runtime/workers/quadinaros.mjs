import { call, fetchBytes, fetchJson, guard, key, load, pick, tick } from "./common.mjs";

const ARRAYS = { float32: Float32Array, int32: Int32Array, uint8: Uint8Array, bool: Uint8Array };

function unpack(bytes, header) {
  return header.order.map((name) => {
    const field = header.fields[name];
    const raw = bytes.slice(field.offset, field.offset + field.bytes);
    if (field.packed) {
      const count = field.shape.reduce((a, b) => a * b, 1);
      const out = new (ARRAYS[field.dtype])(count);
      for (let i = 0; i < count; i++) out[i] = (raw[i >> 3] >> (i & 7)) & 1;
      return out;
    }
    return new (ARRAYS[field.dtype])(raw.buffer);
  });
}

onmessage = guard(async ({ base, seed }) => {
  const manifest = await fetchJson(new URL("manifest.json", base));
  const bytes = await fetchBytes(new URL(manifest.dataset, base));
  const dataset = unpack(bytes, manifest.dataset_header);
  postMessage({ type: "compiling" });
  const init = await load(base, manifest.init);
  const fit = await load(base, manifest.fit);
  const evaluate = await load(base, manifest.evaluate);
  postMessage({ type: "compiled", ms: init.ms + fit.ms + evaluate.ms });

  let state = call(init, [key(seed, 0), ...dataset]);
  postMessage({ type: "params", leaves: pick(state, manifest.params) });
  const updates = manifest.updates_per_call;
  const evaluate0 = call(evaluate, [...state, key(seed ^ 0x2545f491, 0)])[0];
  postMessage({ type: "updates", updates: 0, batchSize: manifest.batch_size, summary: evaluate0, ms: 0 });
  let lastEvaluation = performance.now();
  for (let counter = 1; ; counter++) {
    const started = performance.now();
    state = call(fit, [...state, ...dataset, key(seed, counter)]);
    const ms = performance.now() - started;
    postMessage({ type: "params", leaves: pick(state, manifest.params) });
    let summary = null;
    if (performance.now() - lastEvaluation > manifest.evaluation_every) {
      summary = call(evaluate, [...state, key(seed ^ 0x2545f491, counter)])[0];
      lastEvaluation = performance.now();
    }
    postMessage({ type: "updates", updates, batchSize: manifest.batch_size, summary, ms });
    await tick();
  }
});
