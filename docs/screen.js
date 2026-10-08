export const SCREEN = { width: 192, height: 120 };

const INK = {
  back: "rgb(8, 8, 10)",
  frame: "rgb(40, 40, 40)",
  dim: "rgb(110, 104, 98)",
  text: "rgb(236, 236, 236)",
  red: "rgb(200, 72, 72)",
  orange: "rgb(198, 108, 58)",
  yellow: "rgb(206, 206, 92)",
  green: "rgb(72, 160, 72)",
  blue: "rgb(96, 104, 230)",
  sand: "rgb(118, 88, 58)",
};
const CHANNELS = {
  breakout: [INK.yellow, INK.text, INK.dim, INK.red],
  catch: [INK.yellow, INK.text],
};
const FALLBACK_CHANNELS = [INK.yellow, INK.text, INK.red, INK.orange, INK.blue, INK.green];

function pen(context) {
  const rect = (x, y, w, h, fill) => { context.fillStyle = fill; context.fillRect(Math.round(x), Math.round(y), Math.round(w), Math.round(h)); };
  const line = (x0, y0, x1, y1, fill, thick = 1) => {
    let x = Math.round(x0), y = Math.round(y0);
    const ex = Math.round(x1), ey = Math.round(y1);
    const dx = Math.abs(ex - x), dy = -Math.abs(ey - y);
    const sx = x < ex ? 1 : -1, sy = y < ey ? 1 : -1;
    let error = dx + dy;
    const half = Math.floor(thick / 2);
    for (let guard = 0; guard < 2000; guard++) {
      rect(x - half, y - half, thick, thick, fill);
      if (x === ex && y === ey) break;
      const twice = 2 * error;
      if (twice >= dy) { error += dy; x += sx; }
      if (twice <= dx) { error += dx; y += sy; }
    }
  };
  const text = (value, x, y, fill, align = "left", size = 8) => {
    context.fillStyle = fill;
    context.font = `${size}px "Press Start 2P", monospace`;
    context.textAlign = align;
    context.textBaseline = "top";
    context.fillText(value, Math.round(x), Math.round(y));
  };
  return { rect, line, text };
}

function cartpole(draw, frame, box) {
  const ground = box.y + box.h - 14;
  draw.rect(box.x, ground, box.w, 1, INK.dim);
  for (let x = box.x; x < box.x + box.w; x += 8) draw.rect(x, ground + 3, 4, 1, INK.frame);
  const span = box.w - 24;
  const cx = box.x + 12 + ((Math.max(-2.4, Math.min(2.4, frame.x)) + 2.4) / 4.8) * span;
  draw.rect(box.x + 12, ground - 2, 1, 3, INK.red);
  draw.rect(box.x + box.w - 13, ground - 2, 1, 3, INK.red);
  draw.rect(cx - 13, ground - 11, 26, 9, INK.yellow);
  draw.rect(cx - 12, ground - 12, 24, 1, "rgb(230, 230, 140)");
  draw.rect(cx - 9, ground - 2, 4, 2, INK.dim);
  draw.rect(cx + 5, ground - 2, 4, 2, INK.dim);
  const length = Math.min(56, box.h - 34);
  const tipX = cx + Math.sin(frame.theta) * length;
  const tipY = ground - 12 - Math.cos(frame.theta) * length;
  const lean = Math.min(1, Math.abs(frame.theta) / 0.21);
  draw.line(cx, ground - 12, tipX, tipY, lean > 0.75 ? INK.red : INK.orange, 3);
  draw.rect(cx - 1, ground - 13, 3, 3, INK.text);
}

function grid(draw, frame, box) {
  const { height, width } = frame;
  const channels = frame.channels || 1;
  const colors = CHANNELS[frame.track] || FALLBACK_CHANNELS;
  const cell = Math.max(2, Math.floor(Math.min((box.w - 4) / width, (box.h - 4) / height)));
  const left = box.x + Math.floor((box.w - cell * width) / 2);
  const top = box.y + Math.floor((box.h - cell * height) / 2);
  draw.rect(left - 2, top - 2, cell * width + 4, cell * height + 4, INK.frame);
  draw.rect(left, top, cell * width, cell * height, INK.back);
  for (let r = 0; r < height; r++) {
    for (let c = 0; c < width; c++) {
      let fill = null;
      for (let k = 0; k < channels; k++) {
        if (frame.grid[(r * width + c) * channels + k]) fill = colors[k % colors.length];
      }
      if (fill) draw.rect(left + c * cell + 1, top + r * cell + 1, cell - 2, cell - 2, fill);
      else if (cell >= 6) draw.rect(left + c * cell + Math.floor(cell / 2), top + r * cell + Math.floor(cell / 2), 1, 1, INK.frame);
    }
  }
}

function acrobot(draw, frame, box) {
  const cx = box.x + box.w / 2, cy = box.y + box.h / 2 - 4;
  const link = Math.min(box.h, box.w) / 4.4;
  const x1 = cx + Math.sin(frame.theta1) * link, y1 = cy + Math.cos(frame.theta1) * link;
  const angle = frame.theta1 + frame.theta2;
  const x2 = x1 + Math.sin(angle) * link, y2 = y1 + Math.cos(angle) * link;
  draw.rect(box.x + 8, cy - link, box.w - 16, 1, INK.green);
  draw.line(cx, cy, x1, y1, INK.yellow, 3);
  draw.line(x1, y1, x2, y2, INK.orange, 3);
  draw.rect(cx - 1, cy - 1, 3, 3, INK.text);
}

function unknown(draw, frame, box) {
  const entries = Object.entries(frame).filter(([key, value]) => key !== "track" && typeof value === "number").slice(0, 6);
  entries.forEach(([key, value], index) => draw.text(`${key} ${value.toFixed(2)}`, box.x + 6, box.y + 6 + index * 12, INK.text));
}

export function drawFrame(canvas, frame, label) {
  const context = canvas.getContext("2d");
  const draw = pen(context);
  const { width, height } = canvas;
  draw.rect(0, 0, width, height, INK.back);
  draw.rect(0, 0, width, 13, INK.frame);
  draw.text((label || frame?.track || "").toUpperCase(), 4, 3, INK.text);
  const box = { x: 4, y: 16, w: width - 8, h: height - 20 };
  if (!frame) {
    draw.text("NO SIGNAL", width / 2, height / 2 - 4, INK.dim, "center");
    return;
  }
  if (frame.grid) grid(draw, frame, box);
  else if (typeof frame.theta1 === "number") acrobot(draw, frame, box);
  else if (typeof frame.theta === "number") cartpole(draw, frame, box);
  else unknown(draw, frame, box);
}

export function drawIdle(canvas, lines) {
  const context = canvas.getContext("2d");
  const draw = pen(context);
  draw.rect(0, 0, canvas.width, canvas.height, INK.back);
  draw.rect(0, 0, canvas.width, 13, INK.frame);
  draw.text("BOONTA EVE", 4, 3, INK.text);
  lines.forEach((value, index) => draw.text(value, canvas.width / 2, canvas.height / 2 - 6 * lines.length + index * 14, index ? INK.dim : INK.yellow, "center"));
}

const ICONS = {
  cartpole: { track: "cartpole", x: -0.4, theta: 0.12 },
  breakout: (() => {
    const grid = new Uint8Array(10 * 10 * 4);
    for (let r = 1; r < 4; r++) for (let c = 0; c < 10; c++) if ((r + c) % 5) grid[(r * 10 + c) * 4 + 3] = 1;
    grid[(9 * 10 + 5) * 4] = 1;
    grid[(6 * 10 + 4) * 4 + 1] = 1;
    grid[(7 * 10 + 5) * 4 + 2] = 1;
    return { track: "breakout", grid, height: 10, width: 10, channels: 4 };
  })(),
  catch: (() => {
    const grid = new Uint8Array(10 * 5 * 2);
    grid[(9 * 5 + 2) * 2] = 1;
    grid[(4 * 5 + 1) * 2 + 1] = 1;
    return { track: "catch", grid, height: 10, width: 5, channels: 2 };
  })(),
  acrobot: { track: "acrobot", theta1: 2.4, theta2: 0.8 },
};

export function drawBanner(canvas, track) {
  const icon = ICONS[track.id];
  if (icon) {
    drawFrame(canvas, icon, track.name);
    return;
  }
  const context = canvas.getContext("2d");
  const draw = pen(context);
  draw.rect(0, 0, canvas.width, canvas.height, INK.back);
  draw.rect(0, 0, canvas.width, 13, INK.frame);
  draw.text(track.name.toUpperCase(), 4, 3, INK.text);
  draw.text(track.name.slice(0, 1).toUpperCase(), canvas.width / 2, canvas.height / 2 - 8, INK.yellow, "center", 24);
}

export function drawBoard(canvas, points, solved, colors) {
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  const width = canvas.clientWidth || 300, height = canvas.clientHeight || 140;
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const pad = { left: 38, right: 8, top: 10, bottom: 18 };
  const plot = { x: pad.left, y: pad.top, w: width - pad.left - pad.right, h: height - pad.top - pad.bottom };
  const finite = points.filter((point) => Number.isFinite(point.value));
  const values = finite.map((point) => point.value);
  let low = Math.min(0, ...values), high = Math.max(solved * 1.08, ...values);
  if (high - low < 1e-6) high = low + 1;
  const last = points.length ? points[points.length - 1].update : 1;
  const first = points.length ? points[0].update : 0;
  const span = Math.max(20, last - first);
  const sx = (update) => plot.x + ((update - first) / span) * plot.w;
  const sy = (value) => plot.y + plot.h - ((value - low) / (high - low)) * plot.h;
  context.font = '11px "IBM Plex Mono", ui-monospace, monospace';
  context.textBaseline = "middle";
  context.textAlign = "right";
  context.lineWidth = 1;
  context.strokeStyle = colors.rule;
  context.beginPath();
  context.moveTo(plot.x + 0.5, plot.y);
  context.lineTo(plot.x + 0.5, plot.y + plot.h + 0.5);
  context.lineTo(plot.x + plot.w, plot.y + plot.h + 0.5);
  context.stroke();
  const format = (value) => Math.abs(value) >= 100 ? value.toFixed(0) : Math.abs(value) >= 10 ? value.toFixed(0) : value.toFixed(1);
  context.fillStyle = colors.dim;
  context.fillText(format(low), plot.x - 6, sy(low));
  context.fillText(format(high), plot.x - 6, sy(high) + 4);
  context.textAlign = "left";
  context.fillText(`update ${last}`, plot.x, plot.y + plot.h + 11);
  const solvedY = Math.round(sy(solved)) + 0.5;
  context.strokeStyle = colors.green;
  context.setLineDash([4, 3]);
  context.beginPath();
  context.moveTo(plot.x, solvedY);
  context.lineTo(plot.x + plot.w, solvedY);
  context.stroke();
  context.setLineDash([]);
  context.fillStyle = colors.green;
  context.textAlign = "right";
  context.fillText(`solved ${format(solved)}`, plot.x + plot.w, solvedY - 8);
  context.fillStyle = colors.dot;
  finite.forEach((point) => context.fillRect(Math.round(sx(point.update)) - 1, Math.round(sy(point.value)) - 1, 2, 2));
  if (finite.length > 1) {
    context.strokeStyle = colors.line;
    context.lineWidth = 2;
    context.beginPath();
    let sum = 0;
    const recent = [];
    finite.forEach((point, index) => {
      recent.push(point.value);
      sum += point.value;
      if (recent.length > 8) sum -= recent.shift();
      const x = sx(point.update), y = sy(sum / recent.length);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();
  }
}

export function drawBuilding(canvas, frame, label) {
  const context = canvas.getContext("2d");
  const draw = pen(context);
  const { width, height } = canvas;
  draw.rect(0, 0, width, height, INK.back);
  draw.rect(0, 0, width, 13, INK.frame);
  draw.text(label, 4, 3, INK.text);
  const colors = [INK.red, INK.orange, INK.yellow, INK.blue, INK.green];
  const placed = frame % 9;
  const bx = width / 2 - 24, by = height / 2 + 14;
  for (let i = 0; i < Math.min(placed, 8); i++) {
    const row = Math.floor(i / 3), column = i % 3;
    const x = bx + column * 16 + (row % 2) * 8, y = by - row * 9;
    draw.rect(x, y, 15, 8, colors[i % colors.length]);
    draw.rect(x + 3, y - 2, 3, 2, colors[i % colors.length]);
    draw.rect(x + 9, y - 2, 3, 2, colors[i % colors.length]);
  }
  draw.text(".".repeat(1 + (frame % 3)), width / 2, by + 16, INK.dim, "center");
}
