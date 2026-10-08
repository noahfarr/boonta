import { compile } from "../vendor/whlo/src/index.mjs";

export async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}

export async function fetchText(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.text();
}

export async function fetchBytes(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

export async function load(base, program) {
  const text = await fetchText(new URL(program.file, base));
  const started = performance.now();
  const executable = await compile(text);
  return { executable, program, ms: performance.now() - started };
}

export function call({ executable, program }, args) {
  if (args.length !== program.num_inputs) {
    throw new Error(`${program.file}: ${args.length} arguments, expected ${program.num_inputs}`);
  }
  const inputs = {};
  executable.inputs.forEach((slot, i) => {
    inputs[slot.name] = args[program.kept[i]];
  });
  const out = executable.run(inputs);
  return executable.outputs.map((slot) => out[slot.name]);
}

export const key = (seed, counter) => new Uint32Array([seed >>> 0, counter >>> 0]);

export const pick = (leaves, indices) => indices.map((i) => leaves[i]);

export const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

export class Inbox {
  constructor() {
    this.items = [];
    this.waiting = [];
  }
  push(item) {
    const resolve = this.waiting.shift();
    if (resolve) resolve(item);
    else this.items.push(item);
  }
  next() {
    if (this.items.length) return Promise.resolve(this.items.shift());
    return new Promise((resolve) => this.waiting.push(resolve));
  }
}

export function guard(body) {
  return async (event) => {
    try {
      await body(event.data);
    } catch (error) {
      postMessage({ type: "error", message: String(error?.stack ?? error?.message ?? error) });
    }
  };
}
