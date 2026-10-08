const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const TINTS = ["--red", "--orange", "--yellow"];

function random(key) {
  let hash = 2166136261;
  for (let i = 0; i < key.length; i++) { hash ^= key.charCodeAt(i); hash = Math.imul(hash, 16777619); }
  hash ^= hash >>> 13; hash = Math.imul(hash, 0x5bd1e995); hash ^= hash >>> 15;
  return (hash >>> 0) / 4294967296;
}

function painter(context) {
  const rect = (x, y, w, h, fill) => { context.fillStyle = fill; context.fillRect(Math.round(x), Math.round(y), w, h); };
  const segment = (x0, y0, x1, y1, fill) => {
    let x = Math.round(x0), y = Math.round(y0);
    const ex = Math.round(x1), ey = Math.round(y1);
    const dx = Math.abs(ex - x), dy = -Math.abs(ey - y);
    const sx = x < ex ? 1 : -1, sy = y < ey ? 1 : -1;
    let error = dx + dy;
    for (;;) {
      rect(x, y, 1, 1, fill);
      if (x === ex && y === ey) break;
      const twice = 2 * error;
      if (twice >= dy) { error += dy; x += sx; }
      if (twice <= dx) { error += dx; y += sy; }
    }
  };
  const engine = (x, y, tint, light, flame) => {
    rect(x - 2, y + 1, 2, 3, css("--grey"));
    rect(x, y + 1, 20, 3, tint);
    rect(x + 2, y, 16, 1, light);
    rect(x + 19, y + 1, 1, 3, css("--dark"));
    rect(x + 20, y + 1, 4, 2, tint);
    rect(x + 24, y + 2, 2, 1, light);
    const tones = [css("--text"), css("--yellow"), css("--orange"), css("--red")];
    for (let i = 0; i < flame; i++) rect(x - 3 - i, y + 2, 1, i < flame / 2 ? 2 : 1, tones[Math.min(3, Math.floor((i / flame) * 4))]);
  };
  const pod = (x, y, index, frame) => {
    const tint = css(TINTS[index]), light = css(`${TINTS[index]}-light`);
    const flame = (offset) => 2 + Math.floor(random(`${index}-flame-${offset}-${frame}`) * 6);
    segment(x - 2, y + 2, x - 16, y + 5, css("--grey"));
    segment(x + 1, y + 11, x - 16, y + 7, css("--grey"));
    engine(x, y, tint, light, flame(0));
    engine(x + 3, y + 9, tint, light, flame(1));
    rect(x - 26, y + 4, 10, 5, tint);
    rect(x - 25, y + 3, 7, 1, light);
    rect(x - 20, y + 2, 3, 2, css("--blue"));
    rect(x - 26, y + 2, 2, 2, tint);
    for (let row = y + 4; row < y + 9; row++) rect(x + 9 + Math.floor(random(`${index}-arc-${row}-${frame}`) * 3) - 1, row, 1, 1, frame % 2 ? css("--text") : css("--yellow"));
  };
  return { rect, pod };
}

const sprites = Array.from(document.querySelectorAll("canvas[data-pod]")).map((canvas) => ({ canvas, paint: painter(canvas.getContext("2d")), index: Number(canvas.dataset.pod) }));
function draw(frame) {
  sprites.forEach(({ canvas, paint, index }) => {
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    paint.pod(36, 2, index, frame);
  });
}
draw(0);
const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let visible = false, tick = 0, timer = 0;
const racers = document.querySelector(".racers");
const run = () => {
  clearInterval(timer);
  timer = 0;
  if (visible && !document.hidden && !still) timer = setInterval(() => draw(++tick), 66);
};
if (racers) new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; run(); }).observe(racers);
document.addEventListener("visibilitychange", run);

const copy = document.getElementById("copy");
copy.addEventListener("click", () => {
  const pre = document.getElementById("commands");
  const text = pre.textContent.split("\n").filter((line) => line.startsWith("$ ")).map((line) => line.slice(2)).join("\n");
  const done = () => { copy.textContent = "Copied"; setTimeout(() => { copy.textContent = "Copy"; }, 1600); };
  const fallback = () => {
    const range = document.createRange();
    range.selectNodeContents(pre);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    copy.textContent = "Selected";
  };
  if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, fallback);
  else fallback();
});
