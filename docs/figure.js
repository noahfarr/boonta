import { drawFrame, drawIdle, drawBoard, drawBuilding, drawBanner } from "./screen.js";

async function runtime() {
  if (!new URLSearchParams(location.search).has("mock")) {
    try {
      const module = await import("./runtime/garage.mjs");
      const menu = await module.catalog();
      if (menu && menu.runs && menu.runs.length) return { ...module, demo: false };
    } catch (error) {}
  }
  return { ...(await import("./mock.mjs")), demo: true };
}
const { catalog, start, demo } = await runtime();

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const TINTS = { anakin: "--red", sebulba: "--orange", quadinaros: "--yellow" };
const KINDS = { podracer: "podracers", pilot: "pilots", track: "tracks" };
const hero = $("hero");
const game = $("game");
const screen = $("screen");
const board = $("board");

const menu = await catalog();
const first = menu.runs[0];
const selection = { podracer: first.podracer, pilot: first.pilot, track: first.track };
const buttons = { podracer: {}, pilot: {}, track: {} };
let handle = null, token = 0, points = [], dots = [], buildTimer = 0;

const entry = (kind, id) => menu[KINDS[kind]].find((item) => item.id === id);
const fits = (choice) => menu.runs.find((run) => run.podracer === choice.podracer && run.pilot === choice.pilot && run.track === choice.track);
const allowed = (kind, id) => menu.runs.some((run) => run[kind] === id && (kind === "podracer" || run.podracer === selection.podracer));

function slot(kind, item, parent, content) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = kind === "podracer" ? "pod" : "slot";
  button.setAttribute("role", "radio");
  button.dataset.explain = item.blurb;
  button.dataset.title = item.name;
  button.append(...content);
  button.addEventListener("click", () => choose(kind, item.id));
  buttons[kind][item.id] = button;
  parent.append(button);
}

function build() {
  menu.podracers.forEach((item, index) => {
    if (index) {
      const or = document.createElement("span");
      or.className = "or";
      or.textContent = "or";
      or.setAttribute("aria-hidden", "true");
      $("pods").append(or);
    }
    const sprite = document.createElement("canvas");
    sprite.width = 64;
    sprite.height = 18;
    sprite.dataset.pod = String(index);
    sprite.setAttribute("aria-hidden", "true");
    const name = document.createElement("span");
    name.textContent = item.name.toUpperCase();
    slot("podracer", item, $("pods"), [sprite, name]);
    buttons.podracer[item.id].style.setProperty("--tint", `var(${TINTS[item.id] || "--accent"})`);
  });
  menu.pilots.forEach((item) => slot("pilot", item, $("pilots"), [item.name]));
  menu.tracks.forEach((item) => slot("track", item, $("tracks"), [item.name]));
}

function choose(kind, id) {
  if (handle || !allowed(kind, id)) return;
  selection[kind] = id;
  if (kind === "podracer" && !allowed("pilot", selection.pilot)) {
    selection.pilot = menu.runs.find((run) => run.podracer === id).pilot;
  }
  describeItem(kind, id);
  render();
}

function render() {
  Object.keys(buttons).forEach((kind) => {
    Object.entries(buttons[kind]).forEach(([id, button]) => {
      const chosen = selection[kind] === id;
      const locked = !allowed(kind, id);
      button.setAttribute("aria-checked", String(chosen));
      button.classList.toggle("locked", locked);
      button.disabled = locked || (handle !== null && !chosen);
    });
  });
  const offline = selection.podracer === "quadinaros";
  hero.dataset.offline = String(offline);
  hero.style.setProperty("--tint", `var(${TINTS[selection.podracer] || "--accent"})`);
  $("config").textContent = `podracer=${selection.podracer}\nalgorithm=${selection.pilot}\nenvironment=${selection.track}`;
  $("steps-label").textContent = offline ? "SAMPLES" : "STEPS";
  $("lag-cell").hidden = selection.podracer !== "sebulba";
  if (!handle) drawBanner(game, entry("track", selection.track));
  paintBoard();
}

function explain(title, text) {
  $("caption-title").textContent = title || "";
  $("caption-text").textContent = text || "";
}
const describeItem = (kind, id) => explain(entry(kind, id).name, entry(kind, id).blurb);

function paintBoard() {
  if (document.hidden) return;
  drawBoard(board, points, dots, { rule: css("--rule"), dim: css("--dim"), line: css("--accent"), dot: css("--orange") }, entry("track", selection.track).range);
}

function format(value) {
  if (!Number.isFinite(value)) return "–";
  if (Math.abs(value) >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (Math.abs(value) >= 1e4) return `${(value / 1e3).toFixed(1)}K`;
  if (Math.abs(value) >= 100) return value.toFixed(0);
  return value.toFixed(2).replace(/\.?0+$/, "") || "0";
}

function status(phase, word, detail = "") {
  const node = $("status");
  node.dataset.phase = phase;
  node.textContent = word;
  node.title = detail;
}

function building(current, label) {
  stopBuilding();
  let frame = 0;
  const draw = () => {
    if (current !== token) return stopBuilding();
    if (!document.hidden) drawBuilding(screen, frame++, label);
  };
  draw();
  buildTimer = setInterval(draw, 160);
}
function stopBuilding() { clearInterval(buildTimer); buildTimer = 0; }

function reset() {
  points = [];
  dots = [];
  ["updates", "steps", "lag"].forEach((id) => { $(id).textContent = "0"; });
  $("return").textContent = "–";
}

async function race() {
  const run = fits(selection);
  if (!run) return;
  const current = ++token;
  const choice = { ...selection };
  const label = entry("track", choice.track).name;
  reset();
  hero.dataset.mode = "race";
  $("run").textContent = "■ STOP";
  handle = { stop() {} };
  render();
  building(current, "LOADING");
  status("loading", "Loading");
  if (window.matchMedia("(max-width: 960px)").matches) $("stand").scrollIntoView({ block: "start", behavior: "smooth" });
  const words = { loading: "Loading", compiling: "Compiling", training: "Training", stopped: "Stopped", error: "Error" };
  try {
    const started = await start({ ...choice, seed: 0 }, {
      onStatus(update) {
        if (current !== token) return;
        status(update.phase, words[update.phase] || update.phase, update.phase === "error" ? update.message : "");
        if (update.phase === "loading" || update.phase === "compiling") building(current, update.phase.toUpperCase());
        else stopBuilding();
        if (update.phase === "stopped" || update.phase === "error") finish();
      },
      onUpdate(update) {
        if (current !== token) return;
        if (Number.isFinite(update.episodeReturn)) points.push({ update: update.update, value: update.episodeReturn });
        (update.episodes || []).forEach((value) => dots.push({ update: update.update, value }));
        if (points.length > 20000) points.splice(0, points.length - 20000);
        if (dots.length > 20000) dots.splice(0, dots.length - 20000);
        $("updates").textContent = String(update.update);
        $("steps").textContent = format(update.steps);
        if (Number.isFinite(update.episodeReturn)) $("return").textContent = format(update.episodeReturn);
        if (update.lag !== undefined) $("lag").textContent = String(update.lag);
        paintBoard();
      },
      onFrame(frame) {
        if (current !== token || document.hidden) return;
        drawFrame(screen, frame, label);
      },
    });
    if (current !== token) { started.stop(); return; }
    handle = started;
  } catch (error) {
    if (current !== token) return;
    status("error", "Error", String(error.message || error));
    finish();
  }
}

function finish() {
  handle = null;
  stopBuilding();
  hero.dataset.mode = "idle";
  $("run").textContent = "▶ RUN";
  render();
}

function stop() {
  token += 1;
  if (handle) handle.stop();
  status("stopped", "Stopped");
  finish();
}

function theme() {
  const root = document.documentElement;
  const order = ["auto", "dark", "light"];
  const next = order[(order.indexOf(root.dataset.theme || "auto") + 1) % order.length];
  root.dataset.theme = next;
  const prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
  root.dataset.resolved = next === "auto" ? (prefersLight ? "light" : "dark") : next;
  try { if (next === "auto") localStorage.removeItem("boonta-theme"); else localStorage.setItem("boonta-theme", next); } catch (error) {}
  $("theme").textContent = `THEME ${next.toUpperCase()}`;
  paintBoard();
}

function boot() {
  $("theme").textContent = `THEME ${(document.documentElement.dataset.theme || "auto").toUpperCase()}`;
  $("theme").addEventListener("click", theme);
  if (demo) {
    const badge = document.createElement("span");
    badge.className = "chip demo";
    badge.textContent = "DEMO NUMBERS";
    badge.title = "The training runtime could not be loaded, so these numbers are simulated.";
    document.querySelector(".links").prepend(badge);
  }
  build();
  drawIdle(screen, ["PRESS RUN", "TO WATCH IT LEARN"]);
  render();
  $("run").addEventListener("click", () => (handle ? stop() : race()));
  const figure = $("figure");
  const describe = (event) => {
    const part = event.target.closest("[data-explain]");
    if (part && figure.contains(part)) explain(part.dataset.title, part.dataset.explain);
  };
  figure.addEventListener("pointerover", describe);
  figure.addEventListener("focusin", describe);
  describeItem("podracer", selection.podracer);
  window.matchMedia("(prefers-color-scheme: light)").addEventListener?.("change", () => {
    if (document.documentElement.dataset.theme !== "auto") return;
    document.documentElement.dataset.resolved = window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
    paintBoard();
  });
  window.addEventListener("resize", paintBoard);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) paintBoard(); });
  if (document.fonts) document.fonts.ready.then(paintBoard);
  window.addEventListener("pagehide", () => { if (handle) handle.stop(); });
  window.dispatchEvent(new Event("figure-ready"));
}

boot();
