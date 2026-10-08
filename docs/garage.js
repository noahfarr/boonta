import { createWorld } from "./toyworld.js";
import { drawFrame, drawIdle, drawBoard, drawBuilding } from "./screen.js";

async function runtime() {
  if (!new URLSearchParams(location.search).has("mock")) {
    const paths = /\/ui\/(index\.html)?$/.test(location.pathname) ? ["../runtime/garage.mjs", "./runtime/garage.mjs"] : ["./runtime/garage.mjs", "../runtime/garage.mjs"];
    for (const path of paths) {
      try {
        const module = await import(path);
        const menu = await module.catalog();
        if (menu && menu.runs && menu.runs.length) return { ...module, demo: false };
      } catch (error) {}
    }
  }
  return { ...(await import("./mock.mjs")), demo: true };
}
const { catalog, start, demo } = await runtime();

const KINDS = ["podracer", "pilot", "track"];
const PLURAL = { podracer: "podracers", pilot: "pilots", track: "tracks" };
const TITLES = { podracer: "PILOT", pilot: "POD", track: "TRACK" };
const TINTS = { anakin: "var(--red)", sebulba: "var(--orange)", quadinaros: "var(--yellow)" };
const EXPLAIN = {
  anakin: "Solo run: one program acts and learns. Speed follows steps per second.",
  sebulba: "The pod acts, the infield tower learns. Parameters arrive one update late.",
  quadinaros: "On the dyno: trains from recorded data and never steps the track.",
};

const PARTS = {
  ppo: "Red engine is the actor, blue is the critic. The clamps on the binder are the clip range; the dial is the value estimate.",
  pqn: "Many small engines are the parallel environments. No trailer and no ghost: no replay buffer, no target network.",
  dqn: "The trailer is the replay buffer and the crane samples from it. The ghost is the target network.",
  bc: "The reel feeds recorded demonstrations to the cockpit. In the race it shadows a ghost demonstrator.",
  iql: "The canister holds the dataset and the guard rails keep it inside the data. Gauges show V, Q and the policy.",
};
const $ = (id) => document.getElementById(id);
const garage = $("garage");
const picks = $("picks");
const tags = $("tags");
const screen = $("screen");
const board = $("board");
const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const state = {
  menu: null,
  selection: {},
  touched: [],
  focus: null,
  run: null,
  token: 0,
  points: [],
  frame: null,
  frameDirty: false,
  last: null,
  boardDirty: false,
  phase: "idle",
};
let world = null;
const buttons = {};

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const entry = (kind, id) => state.menu[PLURAL[kind]].find((item) => item.id === id);
const fits = (selection) => state.menu.runs.find((run) => KINDS.every((kind) => run[kind] === selection[kind]));
const STEPS = ["podracer", "pilot", "track"];
let stepIndex = 0;
let busy = false;
const confirmed = {};
const current = () => STEPS[stepIndex];
const allowed = (kind, id) => state.menu.runs.some((run) => STEPS.slice(0, STEPS.indexOf(kind)).every((other) => run[other] === confirmed[other]) && run[kind] === id);
const availability = () => {
  const result = {};
  KINDS.forEach((kind) => {
    result[kind] = {};
    state.menu[PLURAL[kind]].forEach((item) => { result[kind][item.id] = STEPS.indexOf(kind) > stepIndex ? true : allowed(kind, item.id); });
  });
  return result;
};

let lastDescribed = "";
function describe(target) {
  const { kind, id } = target;
  const key = `${kind}:${id}`;
  if (key !== lastDescribed && lastDescribed && !reduced) {
    const card = $("card");
    card.classList.add("swap");
    requestAnimationFrame(() => requestAnimationFrame(() => card.classList.remove("swap")));
  }
  lastDescribed = key;
  const item = entry(kind, id);
  $("card-kind").textContent = { podracer: "TRAINING LOOP", pilot: "ALGORITHM", track: "ENVIRONMENT" }[kind];
  $("card-name").textContent = item.name.toUpperCase();
  $("card-blurb").textContent = kind === "pilot" && PARTS[id] ? `${item.blurb} ${PARTS[id]}` : item.blurb;
  $("card").style.setProperty("--tint", kind === "podracer" ? TINTS[id] || "var(--accent)" : "var(--accent)");
  const ok = allowed(kind, id);
  $("card-warn").textContent = ok ? "" : `Locked: no run for ${STEPS.slice(0, STEPS.indexOf(kind)).map((other) => entry(other, confirmed[other]).name).join(" + ")} with ${item.name}.`;
  $("start").disabled = !ok || busy;
  $("start").classList.toggle("locked", !ok);
  $("start").textContent = !ok ? "\u{1F512} LOCKED" : kind === "track" ? `RACE ${item.name.toUpperCase()}!` : `PICK ${item.name.toUpperCase()}`;
  const line = $("pick-line");
  line.textContent = "";
  STEPS.slice(0, stepIndex).forEach((other) => {
    const chip = document.createElement("span");
    chip.className = "build-chip";
    chip.textContent = entry(other, confirmed[other]).name.toUpperCase();
    line.append(chip);
  });
  const run = kind === "track" && ok ? fits({ ...confirmed, track: id }) : null;
  $("pick-detail").textContent = run ? `${run.num_envs} envs · solved at ${run.solved}` : "";
}

function render() {
  const kind = current();
  const available = availability();
  KINDS.forEach((other) => {
    state.menu[PLURAL[other]].forEach((item) => {
      const button = buttons[`${other}:${item.id}`];
      button.setAttribute("aria-pressed", String(state.selection[other] === item.id));
      button.classList.toggle("off", !available[other][item.id]);
    });
    groups[other].hidden = other !== kind;
  });
  document.querySelectorAll(".steps li").forEach((node, index) => {
    node.dataset.state = index < stepIndex ? "done" : index === stepIndex ? "current" : "next";
    if (index === stepIndex) node.setAttribute("aria-current", "step"); else node.removeAttribute("aria-current");
  });
  $("step-back").disabled = stepIndex === 0 || busy;
  describe({ kind, id: state.selection[kind] });
  if (world) {
    world.setRow(kind);
    world.setSelection(state.selection, available);
    world.setFocus({ kind, id: state.selection[kind] }, document.activeElement === $("viewport"));
  }
}

function focusItem(kind, id, speak = true) {
  if (kind !== current() || busy) return;
  state.selection[kind] = id;
  render();
  if (speak) announce({ kind, id });
}

function cycle(direction) {
  const kind = current();
  const list = state.menu[PLURAL[kind]];
  const index = list.findIndex((item) => item.id === state.selection[kind]);
  focusItem(kind, list[(index + direction + list.length) % list.length].id);
}

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, reduced ? 0 : ms));
async function confirm() {
  const kind = current();
  const id = state.selection[kind];
  if (busy || garage.dataset.mode !== "garage") return;
  if (!allowed(kind, id)) { announce({ kind, id }, "Locked. "); return; }
  confirmed[kind] = id;
  busy = true;
  render();
  if (kind === "podracer") {
    if (world) world.cheer(id);
    $("announce").textContent = `${entry(kind, id).name} is your pilot. Now pick a pod.`;
    await wait(900);
  } else if (kind === "pilot") {
    $("announce").textContent = `${entry(kind, id).name} chosen. ${entry("podracer", confirmed.podracer).name} climbs in. Now pick a track.`;
    if (world) await world.board(confirmed.podracer, id);
    await wait(400);
  } else {
    busy = false;
    state.selection = { ...confirmed };
    race();
    return;
  }
  stepIndex += 1;
  const next = current();
  const list = state.menu[PLURAL[next]];
  const keep = state.selection[next] && allowed(next, state.selection[next]) ? state.selection[next] : (list.find((item) => allowed(next, item.id)) || list[0]).id;
  state.selection[next] = keep;
  busy = false;
  render();
  announce({ kind: next, id: keep });
}

function previous() {
  if (busy || garage.dataset.mode !== "garage" || stepIndex === 0) return;
  stepIndex -= 1;
  const kind = current();
  delete confirmed[kind];
  if (kind === "pilot" && world) world.unboard();
  if (kind === "podracer" && world) world.unboard();
  render();
  announce({ kind, id: state.selection[kind] });
}

const groups = {};
function buildPicks() {
  KINDS.forEach((kind) => {
    const group = document.createElement("div");
    group.className = "group";
    group.setAttribute("role", "group");
    group.setAttribute("aria-labelledby", `legend-${kind}`);
    const legend = document.createElement("span");
    legend.className = "legend";
    legend.id = `legend-${kind}`;
    legend.textContent = TITLES[kind];
    group.append(legend);
    state.menu[PLURAL[kind]].forEach((item) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "pick";
      button.textContent = item.name.toUpperCase();
      button.setAttribute("aria-describedby", "card-blurb");
      button.addEventListener("click", () => focusItem(kind, item.id));
      buttons[`${kind}:${item.id}`] = button;
      group.append(button);
    });
    groups[kind] = group;
    picks.append(group);
  });
}

function place(node, point) {
  if (!point || !point.visible) { node.style.visibility = "hidden"; return; }
  node.style.visibility = "";
  node.style.transform = `translate(${Math.round(point.x - node.offsetWidth / 2)}px, ${Math.round(point.y - node.offsetHeight)}px)`;
}

function onRender(anchors) {
  ["pod", "learner", "reel"].forEach((key) => {
    const node = raceTags[key];
    if (!node) return;
    if (garage.dataset.mode !== "race" || !anchors[key]) { node.hidden = true; return; }
    node.hidden = false;
    place(node, anchors[key]);
  });
}

function announce(target, extra = "") {
  const item = entry(target.kind, target.id);
  const ok = allowed(target.kind, target.id);
  $("announce").textContent = `${TITLES[target.kind].toLowerCase()} ${item.name}${ok ? "" : ", locked"}. ${extra}${item.blurb}`;
}
function keys(event) {
  if (garage.dataset.mode !== "garage") return;
  if (event.key === "ArrowLeft" || event.key === "ArrowUp") cycle(-1);
  else if (event.key === "ArrowRight" || event.key === "ArrowDown") cycle(1);
  else if (event.key === "Enter" || event.key === " ") confirm();
  else if (event.key === "Escape" || event.key === "Backspace") previous();
  else return;
  event.preventDefault();
}

const raceTags = {};
function buildRaceTags() {
  [["pod", "var(--text)"], ["learner", "var(--orange-light)"], ["reel", "var(--yellow-light)"]].forEach(([key, tint]) => {
    const node = document.createElement("div");
    node.className = "tag";
    node.style.setProperty("--tint", tint);
    node.hidden = true;
    tags.append(node);
    raceTags[key] = node;
  });
}

function paintBoard() {
  state.boardDirty = false;
  if (document.hidden) { state.boardDirty = true; return; }
  const run = fits(state.selection);
  drawBoard(board, state.points, run ? run.solved : 1, { rule: css("--rule"), dim: css("--dim"), green: css("--green"), line: css("--accent"), dot: css("--orange") });
}

let boardFrame = 0;
function scheduleBoard() {
  state.boardDirty = true;
  if (boardFrame || document.hidden) return;
  boardFrame = requestAnimationFrame(() => { boardFrame = 0; if (state.boardDirty) paintBoard(); });
}

function format(value) {
  if (!Number.isFinite(value)) return "–";
  if (Math.abs(value) >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (Math.abs(value) >= 1e4) return `${(value / 1e3).toFixed(1)}K`;
  if (Math.abs(value) >= 100) return value.toFixed(0);
  return value.toFixed(2).replace(/\.?0+$/, "") || "0";
}

function status(phase, message) {
  state.phase = phase;
  const node = $("status");
  node.dataset.phase = phase;
  node.textContent = message;
  $("stop").disabled = phase === "stopped" || phase === "error";
}

let buildTimer = 0;
function building(token, label) {
  stopBuilding();
  let frame = 0;
  const draw = () => {
    if (token !== state.token) return stopBuilding();
    if (!document.hidden) { drawBuilding(screen, frame++, label); if (world) world.screenChanged(); }
  };
  draw();
  buildTimer = setInterval(draw, 160);
}
function stopBuilding() { clearInterval(buildTimer); buildTimer = 0; }

async function race() {
  const selection = { ...confirmed, ...state.selection };
  const run = fits(selection);
  if (!run) return;
  const token = ++state.token;
  state.points = [];
  state.frame = null;
  state.last = null;
  garage.dataset.mode = "race";
  garage.style.setProperty("--tint", TINTS[selection.podracer] || "var(--accent)");
  $("run-name").textContent = KINDS.map((kind) => entry(kind, selection[kind]).name.toUpperCase()).join(" · ");
  $("explain").textContent = EXPLAIN[selection.podracer] || "";
  const head = document.querySelector(".hud .head");
  head.classList.remove("brief");
  clearTimeout(state.briefTimer);
  state.briefTimer = setTimeout(() => head.classList.add("brief"), 6000);
  const offline = selection.podracer === "quadinaros";
  $("unit").textContent = offline ? "SAMPLES / S" : "STEPS / S";
  $("steps-label").textContent = offline ? "SAMPLES" : "STEPS";
  $("extra-label").textContent = selection.podracer === "sebulba" ? "LAG" : offline ? "TRACK" : "LAP";
  $("extra").textContent = offline ? "idle" : "0";
  ["sps", "updates", "steps"].forEach((id) => { $(id).textContent = "0"; });
  $("return").textContent = "–";
  $("gauge").style.width = "0%";
  $("solved-flag").textContent = "";
  $("camera").textContent = "CAM: AUTO";
  building(token, "LOADING");
  if (garage.classList.contains("stacked")) (world ? $("viewport") : $("hud")).scrollIntoView({ block: "start", behavior: reduced ? "auto" : "smooth" });
  if (world) { world.screenChanged(); world.race(selection); $("camera").textContent = `CAM: ${world.view === "chase" ? "CHASE" : "AUTO"}`; }
  paintBoard();
  status("loading", "Loading");
  const label = entry("track", selection.track).name;
  try {
    const handle = await start({ ...selection, seed: 0 }, {
      onStatus(update) {
        if (token !== state.token) return;
        const seconds = update.compileMs ? ` · compiled in ${(update.compileMs / 1000).toFixed(1)}s` : "";
        const words = { loading: "Loading", compiling: "Compiling", training: "Training", stopped: "Stopped", error: "Error" };
        status(update.phase, `${words[update.phase] || update.phase}${update.phase === "training" ? seconds : update.message ? `: ${update.message}` : ""}`);
        if (world) world.setTraining(update.phase === "training");
        if (update.phase === "loading" || update.phase === "compiling") building(token, update.phase === "loading" ? "LOADING" : "COMPILING");
        else stopBuilding();
        if (update.phase === "stopped" && world) world.setSpeed(0);
        if (world) world.screenChanged();
      },
      onUpdate(update) {
        if (token !== state.token) return;
        state.last = update;
        state.points.push({ update: update.update, value: update.episodeReturn });
        if (state.points.length > 2000) state.points.splice(0, state.points.length - 2000);
        $("sps").textContent = format(update.sps);
        $("updates").textContent = String(update.update);
        $("steps").textContent = format(update.steps);
        if (Number.isFinite(update.episodeReturn)) $("return").textContent = format(update.episodeReturn);
        const level = Math.max(0, Math.min(1, (Math.log10(Math.max(1, update.sps)) - 2.5) / 2.5));
        $("gauge").style.width = `${(level * 100).toFixed(1)}%`;
        if (selection.podracer === "sebulba") $("extra").textContent = String(update.lag ?? 1);
        else if (!offline && world) $("extra").textContent = String(world.laps);
        const recent = state.points.slice(-8).map((point) => point.value).filter(Number.isFinite);
        const mean = recent.reduce((sum, value) => sum + value, 0) / Math.max(1, recent.length);
        if (recent.length >= 4 && mean >= run.solved) $("solved-flag").textContent = "SOLVED";
        if (raceTags.pod) {
          const name = `${entry("podracer", selection.podracer).name.toUpperCase()} · ${entry("pilot", selection.pilot).name.toUpperCase()}`;
          raceTags.pod.innerHTML = selection.podracer === "sebulba"
            ? `${name} · ACTOR<small>params of update ${Math.max(0, update.update - (update.lag ?? 1))}</small>`
            : offline ? `${name}<small>on the dyno</small>` : `${name}<small>${format(update.sps)} steps/s</small>`;
          if (raceTags.learner) raceTags.learner.innerHTML = `LEARNER<small>update ${update.update}</small>`;
          if (raceTags.reel) raceTags.reel.innerHTML = `RECORDED DATA<small>${format(update.steps)} samples</small>`;
        }
        if (world) { world.setSpeed(update.sps); world.pulse(update); }
        scheduleBoard();
      },
      onFrame(frame) {
        if (token !== state.token || document.hidden) return;
        drawFrame(screen, frame, label);
        if (world) world.screenChanged();
      },
    });
    if (token !== state.token) { handle.stop(); return; }
    state.run = handle;
  } catch (error) {
    if (token !== state.token) return;
    status("error", `Error: ${error.message || error}`);
  }
}

function stop() {
  if (state.run) { state.run.stop(); state.run = null; }
}

async function back() {
  state.token += 1;
  stopBuilding();
  stop();
  $("card").classList.add("moving");
  garage.dataset.mode = "garage";
  Object.values(raceTags).forEach((node) => { node.hidden = true; });
  if (world) await world.back();
  render();
  $("card").classList.remove("moving");
  $("viewport").focus({ preventScroll: true });
}

function layout() {
  const width = window.innerWidth, height = window.innerHeight;
  const side = height < 520 && width > height && width >= 560;
  const narrow = !side && width < 720;
  garage.dataset.layout = !world ? "list" : side ? "side" : narrow ? "compact" : "scene";
  garage.classList.toggle("stacked", !world || narrow);
  const back = $("step-back");
  const home = garage.classList.contains("stacked") || side ? document.querySelector(".summary") : document.querySelector(".stepbar");
  if (back.parentNode !== home) home.prepend(back);
  if (world) world.resize();
}

function insets() {
  const layout = garage.dataset.layout;
  if (garage.dataset.mode !== "garage" || (layout !== "scene" && layout !== "side")) return { top: 0, bottom: 0, right: 0 };
  const view = $("viewport"), card = $("card"), bar = document.querySelector(".stepbar");
  const top = bar.offsetTop + bar.offsetHeight + 8;
  if (layout === "side") return { top, bottom: 28, right: view.clientWidth - card.offsetLeft + 8 };
  return { top, bottom: view.clientHeight - card.offsetTop + 12, right: 0 };
}

function theme() {
  const root = document.documentElement;
  const order = ["auto", "dark", "light"];
  const next = order[(order.indexOf(root.dataset.theme || "auto") + 1) % order.length];
  root.dataset.theme = next;
  const prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
  root.dataset.resolved = next === "auto" ? (prefersLight ? "light" : "dark") : next;
  try { if (next === "auto") localStorage.removeItem("boonta-theme"); else localStorage.setItem("boonta-theme", next); } catch (error) {}
  $("theme").textContent = `THEME: ${next.toUpperCase()}`;
  if (world) world.theme();
  paintBoard();
}

async function boot() {
  $("theme").textContent = `THEME: ${(document.documentElement.dataset.theme || "auto").toUpperCase()}`;
  $("theme").addEventListener("click", theme);
  state.menu = await catalog();
  if (demo) {
    const lede = document.querySelector(".lede");
    if (lede) lede.innerHTML = 'Distributed reinforcement learning in JAX, built on the <a href="https://arxiv.org/abs/2104.06272">Podracer architectures</a>. Pick a pilot, a pod and a track above to watch a training race. This preview uses demo numbers; the full site trains in your browser.';
    const badge = document.createElement("span");
    badge.className = "chip demo";
    badge.textContent = "DEMO NUMBERS";
    badge.title = "The training runtime could not be loaded, so these numbers are simulated.";
    document.querySelector(".links").prepend(badge);
  }
  const first = state.menu.runs[0];
  state.selection = { podracer: first.podracer, pilot: first.pilot, track: first.track };
  buildPicks();
  buildRaceTags();
  drawIdle(screen, ["PICK A RACE", "AND PRESS START"]);
  try {
    world = createWorld($("world"), {
      menu: state.menu,
      screen,
      reduced,
      onPick: (kind, id) => {
        if (kind !== current()) return;
        if (state.selection[kind] === id) confirm();
        else focusItem(kind, id);
      },
      onHover: (target) => { if (target && target.kind === current() && target.id !== state.selection[target.kind]) focusItem(target.kind, target.id, false); },
      onMove: (phase) => { $("card").classList.toggle("moving", phase === "start"); },
      onRender,
      insets,
      onLost: () => { world = null; garage.classList.add("flat"); layout(); },
    });
  } catch (error) {
    world = null;
  }
  if (!world) garage.classList.add("flat");
  if (world && new URLSearchParams(location.search).has("check")) window.__probes = world.probes();
  layout();
  render();
  window.addEventListener("resize", layout);
  $("start").addEventListener("click", confirm);
  $("step-back").addEventListener("click", previous);
  const viewport = $("viewport");
  viewport.addEventListener("keydown", keys);
  viewport.addEventListener("focus", () => { if (garage.dataset.mode === "garage" && world) world.setFocus({ kind: current(), id: state.selection[current()] }, true); });
  viewport.addEventListener("blur", () => { if (world) world.setFocus({ kind: current(), id: state.selection[current()] }, false); });
  $("item-prev").addEventListener("click", () => cycle(-1));
  $("item-next").addEventListener("click", () => cycle(1));
  let swipe = null;
  viewport.addEventListener("pointerdown", (event) => { swipe = event.pointerType !== "mouse" && event.isPrimary && garage.dataset.mode === "garage" ? { x: event.clientX, y: event.clientY } : null; });
  viewport.addEventListener("pointercancel", () => { swipe = null; });
  viewport.addEventListener("pointerup", (event) => {
    if (!swipe) return;
    const dx = event.clientX - swipe.x, dy = event.clientY - swipe.y;
    swipe = null;
    if (garage.dataset.mode === "garage" && Math.abs(dx) > 40 && Math.abs(dx) > Math.abs(dy) * 1.5) cycle(dx < 0 ? 1 : -1);
  });
  document.addEventListener("keydown", (event) => {
    if (garage.dataset.mode !== "garage" || event.target === viewport) return;
    if ((event.key === "Escape") && garage.contains(document.activeElement)) { previous(); event.preventDefault(); }
  });
  $("stop").addEventListener("click", stop);
  $("back").addEventListener("click", back);
  $("camera").addEventListener("click", () => {
    if (!world) return;
    const views = ["auto", "chase", "screen", "track"];
    const names = { auto: "AUTO", chase: "CHASE", screen: "SCREEN", track: "TRACK" };
    const next = views[(views.indexOf(world.view) + 1) % views.length];
    world.setView(next);
    $("camera").textContent = `CAM: ${names[next]}`;
  });
  window.matchMedia("(prefers-color-scheme: light)").addEventListener?.("change", () => {
    if (document.documentElement.dataset.theme !== "auto") return;
    document.documentElement.dataset.resolved = window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
    if (world) world.theme();
    paintBoard();
  });
  document.addEventListener("visibilitychange", () => { if (!document.hidden && state.boardDirty) paintBoard(); });
  if (document.fonts) document.fonts.ready.then(() => { if (world) world.refreshText(); paintBoard(); });
  window.addEventListener("pagehide", stop);
}

boot();
