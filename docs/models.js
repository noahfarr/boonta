import { P, BRICK, PLATE, STUD, builder, brickGeometry, faceTexture, grilleTexture, bake } from "./bricks.js";

const T = () => window.THREE;
const GAP = 0.01;

function along(geometry) {
  geometry.rotateZ(-Math.PI / 2);
  return geometry;
}

function mesh(geometry, material, x = 0, y = 0, z = 0, cast = true) {
  const made = new (T().Mesh)(geometry, material);
  made.position.set(x, y, z);
  made.castShadow = cast;
  made.receiveShadow = true;
  return made;
}

function wedge(w, d, h) {
  const THREE = T();
  const shape = new THREE.Shape();
  shape.moveTo(-w / 2, -d / 2);
  shape.lineTo(w / 2, -d / 2);
  shape.lineTo(-w / 2, d / 2);
  shape.lineTo(-w / 2, -d / 2);
  const geometry = new THREE.ExtrudeGeometry(shape, { depth: h, bevelEnabled: false });
  geometry.rotateX(-Math.PI / 2);
  return geometry;
}

function slope(w, d, h) {
  const THREE = T();
  const shape = new THREE.Shape();
  shape.moveTo(-d / 2, 0);
  shape.lineTo(d / 2, 0);
  shape.lineTo(d / 2, PLATE);
  shape.lineTo(-d / 2 + P * 0.5, h);
  shape.lineTo(-d / 2, h);
  shape.lineTo(-d / 2, 0);
  const geometry = new THREE.ExtrudeGeometry(shape, { depth: w - GAP, bevelEnabled: true, bevelThickness: 0.02, bevelSize: 0.02, bevelSegments: 1 });
  geometry.translate(0, 0, -(w - GAP) / 2);
  return geometry;
}

function rod(material, from, to, radius) {
  const THREE = T();
  const direction = new THREE.Vector3().subVectors(to, from);
  const made = mesh(new THREE.CylinderGeometry(radius, radius, direction.length(), 6), material);
  made.position.copy(from).addScaledVector(direction, 0.5);
  made.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
  return made;
}

const grille = { map: null };

function engine(kit, group, kitBuild, spec, materials) {
  const THREE = T();
  const { trim, dark, steel, flame } = materials;
  const body = spec.paint ? kit.mat(spec.paint, { owner: materials.owner }) : materials.body;
  const radius = spec.size === 4 ? 2 * P - 0.02 : spec.size === 1 ? P / 2 - 0.01 : P - 0.01;
  const sections = spec.sections;
  const length = sections * BRICK;
  const part = new THREE.Group();
  part.position.set(spec.x, spec.y, spec.z);
  part.rotation.y = spec.yaw || 0;
  for (let i = 0; i < sections; i++) {
    const tint = i === 1 || i === sections - 2 ? trim : body;
    part.add(mesh(along(new THREE.CylinderGeometry(radius, radius, BRICK - GAP, 24)), tint, -length / 2 + (i + 0.5) * BRICK));
  }
  const front = length / 2;
  part.add(mesh(along(new THREE.CylinderGeometry(radius * 0.98, radius, PLATE - GAP, 24)), dark, front + PLATE / 2));
  if (!grille.map) grille.map = grilleTexture();
  const disc = mesh(new THREE.CircleGeometry(radius * 0.86, 24), kit.raw(new THREE.MeshStandardMaterial({ map: grille.map, roughness: 0.4 })), front + PLATE + 0.002);
  disc.rotation.y = Math.PI / 2;
  part.add(disc);
  part.add(mesh(along(new THREE.ConeGeometry(radius * 0.5, radius * 1.1, 20)), steel, front + PLATE + radius * 0.55));
  part.add(mesh(along(new THREE.CylinderGeometry(STUD.radius, STUD.radius, STUD.height, 12)), steel, front + PLATE + radius * 1.1 + STUD.height / 2));
  part.add(mesh(along(new THREE.CylinderGeometry(radius * 0.8, radius * 0.9, PLATE - GAP, 24)), dark, -front - PLATE / 2));
  const flames = [];
  [[radius * 0.62, radius * 1.6], [radius * 0.36, radius * 2.4]].forEach(([r, l]) => {
    const geometry = new THREE.ConeGeometry(r, l, 16, 1, true);
    geometry.translate(0, l / 2, 0);
    geometry.rotateZ(Math.PI / 2);
    const cone = mesh(geometry, flame, -front - PLATE, 0, 0, false);
    part.add(cone);
    flames.push(cone);
  });
  const mount = spec.size === 1 ? 0 : Math.max(2, sections - 2);
  for (let i = 0; i < mount; i++) {
    const x = -((mount - 1) / 2) * P + i * P;
    part.add(mesh(brickGeometry(P - GAP, P * 2 - GAP, PLATE - GAP, 0.02), trim, x, radius + PLATE / 2 - 0.05, 0));
    part.add(mesh(new THREE.CylinderGeometry(STUD.radius, STUD.radius, STUD.height, 12), trim, x, radius + PLATE - 0.05 + STUD.height / 2, P / 2));
    part.add(mesh(new THREE.CylinderGeometry(STUD.radius, STUD.radius, STUD.height, 12), trim, x, radius + PLATE - 0.05 + STUD.height / 2, -P / 2));
  }
  if (spec.size !== 1) {
    const fin = mesh(wedge(P * 3, P * 2, PLATE - GAP), trim, -front + P * 1.2, -PLATE / 2, radius + P);
    fin.rotation.y = Math.PI;
    part.add(fin);
    const fin2 = mesh(wedge(P * 3, P * 2, PLATE - GAP), trim, -front + P * 1.2, -PLATE / 2, -radius - P);
    fin2.scale.z = -1;
    fin2.rotation.y = Math.PI;
    part.add(fin2);
  }
  if (spec.scoop) {
    for (const side of [-1, 1]) part.add(mesh(brickGeometry(3 * P - GAP, P - GAP, BRICK * 1.4, 0.025), trim, front - P * 2.5, 0, side * (radius + P / 2)));
    part.add(mesh(brickGeometry(4 * P - GAP, 2 * P - GAP, PLATE - GAP, 0.02), dark, 0, -radius - PLATE / 2 + 0.05, 0));
  }
  group.add(part);
  return { flames, rear: spec.x - front - PLATE, front: spec.x + front, radius, part };
}

function head(kit, skin, helmetMaterial, scale = 1, style = "dome", visor = null) {
  const THREE = T();
  const group = new THREE.Group();
  const face = faceTexture(skin.color.clone().convertLinearToSRGB().getStyle());
  kit.onTheme(() => face.draw(skin.color.clone().convertLinearToSRGB().getStyle()));
  const skull = mesh(new THREE.CylinderGeometry(0.39, 0.39, BRICK - GAP, 24, 1, false, Math.PI, Math.PI * 2), kit.raw(new THREE.MeshPhysicalMaterial({ map: face.map, roughness: 0.3, clearcoat: 0.3 })), 0, BRICK / 2);
  group.add(skull);
  if (style === "cap") {
    group.add(mesh(new THREE.CylinderGeometry(0.56, 0.6, PLATE * 1.4, 24), helmetMaterial, 0, BRICK * 0.82));
    group.add(mesh(brickGeometry(1.3, 0.5, 0.08, 0.02), helmetMaterial, 0, BRICK * 0.62, 0.42));
    group.add(mesh(new THREE.CylinderGeometry(STUD.radius, STUD.radius, STUD.height, 12), helmetMaterial, 0, BRICK * 0.82 + PLATE * 0.7 + STUD.height / 2));
  } else {
    const helmet = mesh(new THREE.SphereGeometry(0.47, 24, 12, 0, Math.PI * 2, 0, Math.PI * 0.5), helmetMaterial, 0, BRICK * 0.62);
    helmet.scale.y = style === "crest" ? 1.25 : 0.95;
    group.add(helmet);
    group.add(mesh(new THREE.CylinderGeometry(0.48, 0.48, PLATE * 0.5, 24), helmetMaterial, 0, BRICK * 0.62));
    if (style === "crest") group.add(mesh(brickGeometry(0.16, 1.0, 0.5, 0.02), helmetMaterial, 0, BRICK * 0.62 + 0.62, -0.05));
    else group.add(mesh(new THREE.CylinderGeometry(STUD.radius, STUD.radius, STUD.height, 12), helmetMaterial, 0, BRICK * 0.62 + 0.44 + STUD.height / 2));
  }
  if (visor) {
    const band = mesh(new THREE.CylinderGeometry(0.4, 0.4, 0.2, 24, 1, true, -Math.PI * 0.35, Math.PI * 0.7), visor, 0, BRICK * 0.62, 0);
    band.scale.set(1.02, 1, 1.02);
    group.add(band);
  }
  group.scale.setScalar(scale);
  return group;
}

export function buildPod(kit, design, owner, tint) {
  const THREE = T();
  design = { ...design, tint };
  const body = kit.mat(design.tint, { owner });
  const trim = kit.mat(`${design.tint}-light`, { owner });
  const dark = kit.mat("--shadow3d", { owner });
  const steel = kit.mat("--steel3d", { owner, rough: 0.25 });
  const flame = kit.trans("--orange", { owner, glow: 0.8, opacity: 0.7 });
  const field = kit.trans(design.binder, { owner, glow: 0.9, opacity: 0.6 });
  const clear = kit.trans("--canopy3d", { owner, opacity: 0.35 });
  const skin = kit.mat("--skin3d", { owner });
  const group = new THREE.Group();
  const build = builder(group);
  const materials = { body, trim, dark, steel, flame, owner };
  const extras = { clamps: [], bars: [], gauges: [], stack: [], needle: null, crane: null, reel: null, kind: design.kind };
  const extraParts = [];
  const flames = [];
  const engines = design.engines.map((spec) => {
    const built = engine(kit, group, build, { ...spec, y: 0 }, materials);
    flames.push(...built.flames);
    return { ...spec, ...built };
  });
  const binders = [];
  design.binders.forEach(([a, b]) => {
    const left = engines[a], right = engines[b];
    const x = Math.min(left.front, right.front) - P * 1.5;
    const z0 = left.z + Math.sign(right.z - left.z) * left.radius, z1 = right.z - Math.sign(right.z - left.z) * right.radius;
    const bar = mesh(new THREE.CylinderGeometry(0.16, 0.16, Math.abs(z1 - z0), 8), field, x, 0, (z0 + z1) / 2, false);
    bar.rotation.x = Math.PI / 2;
    const sheet = mesh(new THREE.PlaneGeometry(Math.abs(z1 - z0), P * 0.9), field, x, 0, (z0 + z1) / 2, false);
    sheet.rotation.y = Math.PI / 2;
    group.add(bar, sheet);
    binders.push(sheet, bar);
    if (design.kind === "ppo") {
      for (const z of [z0, z1]) {
        const clamp = mesh(new THREE.CylinderGeometry(0.34, 0.34, 0.36, 16), kit.trans("--yellow-light", { opacity: 0.85 }), x, 0, z, false);
        clamp.rotation.x = Math.PI / 2;
        group.add(clamp);
        extras.clamps.push(clamp);
        extraParts.push(clamp);
      }
    }
  });
  const pit = new THREE.Group();
  const width = design.cockpit.studs * P;
  const depth = 3 * P;
  pit.add(mesh(brickGeometry(width - GAP, depth - GAP, PLATE - GAP, 0.02), body, 0, -BRICK / 2 + PLATE / 2, 0));
  pit.add(mesh(brickGeometry(width - GAP, P - GAP, BRICK - GAP, 0.025), body, 0, PLATE / 2 + 0.0, -P));
  pit.add(mesh(brickGeometry(width - GAP, P - GAP, BRICK - GAP, 0.025), body, 0, PLATE / 2, P));
  pit.add(mesh(brickGeometry(P - GAP, P - GAP, BRICK - GAP, 0.025), trim, -width / 2 + P / 2, PLATE / 2, 0));
  const noseLength = (design.nose || 2) * P;
  const nose = mesh(slope(depth, noseLength, BRICK), trim, width / 2 + noseLength / 2, -BRICK / 2 + PLATE, 0);
  nose.rotation.y = Math.PI;
  pit.add(nose);
  for (let i = 0; i < design.cockpit.studs; i++) {
    const x = -width / 2 + P / 2 + i * P;
    pit.add(mesh(new THREE.CylinderGeometry(STUD.radius, STUD.radius, STUD.height, 12), body, x, PLATE / 2 + BRICK / 2 + STUD.height / 2, P));
    pit.add(mesh(new THREE.CylinderGeometry(STUD.radius, STUD.radius, STUD.height, 12), body, x, PLATE / 2 + BRICK / 2 + STUD.height / 2, -P));
  }
  const canopy = mesh(new THREE.BoxGeometry(0.06, BRICK * 0.9, depth - P * 0.4), clear, width / 2 - 0.1, BRICK * 0.85, 0, false);
  canopy.rotation.z = 0.45;
  pit.add(canopy);
  const helmet = kit.mat("--text", { owner });
  const pilot = head(kit, skin, helmet, 0.9);
  pilot.position.set(-0.1, -BRICK / 2 + PLATE, 0);
  pilot.rotation.y = Math.PI / 2;
  pit.add(pilot);
  if (design.fin) {
    const fin = mesh(slope(P - GAP, 4 * P, 2.5 * BRICK), trim, -width / 2 + P * 0.5, PLATE / 2 + BRICK / 2, 0);
    fin.rotation.y = Math.PI / 2;
    pit.add(fin);
    const tape = mesh(new THREE.PlaneGeometry(2.4 * P, 0.5), kit.trans("--yellow-light", { glow: 0.6, opacity: 0.9 }), -width / 2 + P * 0.3, PLATE / 2 + BRICK * 1.6, P / 2 + 0.01, false);
    pit.add(tape);
  }
  pit.position.set(design.cockpit.x, 0, 0);
  group.add(pit);
  const cable = kit.mat("--shadow3d", { owner, rough: 0.5 });
  const hitch = new THREE.Vector3(design.cockpit.x + width / 2 + noseLength, 0, 0);
  design.cables.forEach(([index, spread]) => {
    const spec = engines[index];
    group.add(rod(cable, new THREE.Vector3(spec.rear + 0.4, 0.15, spec.z), hitch.clone().add(new THREE.Vector3(0, 0.15, spread)), 0.06));
    group.add(rod(cable, new THREE.Vector3(spec.rear + 0.4, -0.2, spec.z), hitch.clone().add(new THREE.Vector3(0, -0.1, spread)), 0.05));
  });
  (design.links || []).forEach(([a, b]) => {
    const from = engines[a], to = engines[b];
    group.add(rod(cable, new THREE.Vector3(from.rear, 0, from.z), new THREE.Vector3(to.front - 0.3, 0, to.z), 0.08));
  });
  const top = PLATE / 2 + BRICK / 2;
  const cx = design.cockpit.x;
  const add = (made) => { group.add(made); extraParts.push(made); return made; };
  const letter = (text, color) => {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 64;
    const paint = canvas.getContext("2d");
    paint.fillStyle = "#151515";
    paint.fillRect(0, 0, 64, 64);
    paint.fillStyle = color;
    paint.font = "bold 40px IBM Plex Mono, monospace";
    paint.textAlign = "center";
    paint.textBaseline = "middle";
    paint.fillText(text, 32, 34);
    return new THREE.CanvasTexture(canvas);
  };
  if (design.kind === "ppo") {
    const dialCanvas = document.createElement("canvas");
    dialCanvas.width = dialCanvas.height = 64;
    const dialPaint = dialCanvas.getContext("2d");
    dialPaint.fillStyle = "#f2f2f2";
    dialPaint.beginPath(); dialPaint.arc(32, 32, 31, 0, Math.PI * 2); dialPaint.fill();
    dialPaint.strokeStyle = "#202020";
    dialPaint.lineWidth = 3;
    for (let i = 0; i <= 8; i++) {
      const a = Math.PI * 0.75 + (i / 8) * Math.PI * 1.5;
      dialPaint.beginPath(); dialPaint.moveTo(32 + Math.cos(a) * 22, 32 + Math.sin(a) * 22); dialPaint.lineTo(32 + Math.cos(a) * 29, 32 + Math.sin(a) * 29); dialPaint.stroke();
    }
    const width = design.cockpit.studs * P;
    add(mesh(new THREE.CylinderGeometry(P - 0.02, P - 0.02, 0.1, 24), dark, cx - width / 2 + P, top + 0.05, 0));
    const face = add(mesh(new THREE.CircleGeometry(P - 0.08, 24), kit.raw(new THREE.MeshPhysicalMaterial({ map: new THREE.CanvasTexture(dialCanvas), roughness: 0.15, clearcoat: 0.6 })), cx - width / 2 + P, top + 0.11, 0, false));
    face.rotation.x = -Math.PI / 2;
    const needle = new THREE.Group();
    needle.position.set(cx - width / 2 + P, top + 0.13, 0);
    needle.add(mesh(new THREE.BoxGeometry(0.06, 0.03, P * 0.8), kit.mat("--red", { owner }), 0, 0, -P * 0.35, false));
    add(needle);
    extras.needle = needle;
  }
  if (design.kind === "pqn") {
    const width = design.cockpit.studs * P;
    for (let i = 0; i < 4; i++) {
      const bar = new THREE.Group();
      bar.position.set(cx - width / 2 + P * 0.6, top, (i - 1.5) * 0.42);
      bar.add(mesh(new THREE.BoxGeometry(0.3, 1, 0.3).translate(0, 0.5, 0), kit.trans(i % 2 ? "--green" : "--yellow-light", { glow: 0.8, opacity: 0.9 }), 0, 0, 0, false));
      bar.scale.y = 0.3 + 0.2 * i;
      add(bar);
      extras.bars.push(bar);
    }
  }
  if (design.kind === "dqn") {
    const width = design.cockpit.studs * P;
    const trailer = new THREE.Group();
    trailer.position.set(cx - width / 2 - 3.6 * P - 0.8, -BRICK / 2, 0);
    trailer.add(mesh(brickGeometry(6 * P - GAP, 4 * P - GAP, PLATE - GAP, 0.02), dark, 0, PLATE / 2, 0));
    const memory = ["--blue", "--green", "--yellow-light", "--orange-light"];
    for (let level = 0; level < 4; level++) for (let row = 0; row < 3; row++) for (let column = 0; column < 2; column++) {
      const brick = mesh(brickGeometry(2 * P - GAP, P - GAP, BRICK - GAP, 0.025), kit.mat(memory[(level + row + column) % memory.length], { owner }), (row - 1) * 2 * P, PLATE + BRICK / 2 + level * BRICK, (column - 0.5) * 2 * P);
      brick.rotation.y = Math.PI / 2;
      trailer.add(brick);
      extras.stack.push(brick);
    }
    const crane = new THREE.Group();
    crane.position.set(2.6 * P, PLATE, 1.6 * P);
    crane.add(mesh(brickGeometry(P - GAP, P - GAP, 4 * BRICK - GAP, 0.02), kit.mat("--yellow", { owner }), 0, 2 * BRICK, 0));
    const boom = new THREE.Group();
    boom.position.y = 4 * BRICK;
    boom.add(mesh(new THREE.CylinderGeometry(0.16, 0.16, 4 * P, 8).rotateZ(Math.PI / 2), kit.mat("--yellow", { owner }), -1.6 * P, 0, 0));
    boom.add(mesh(new THREE.CylinderGeometry(0.03, 0.03, 1.4, 6), dark, -3.4 * P, -0.7, 0));
    boom.add(mesh(new THREE.ConeGeometry(0.2, 0.3, 8), steel, -3.4 * P, -1.45, 0));
    crane.add(boom);
    trailer.add(crane);
    extras.crane = boom;
    add(trailer);
    add(rod(dark, new THREE.Vector3(cx - width / 2, -0.2, 0), new THREE.Vector3(trailer.position.x + 3 * P, -0.2, 0), 0.1));
  }
  if (design.kind === "bc") {
    const width = design.cockpit.studs * P;
    const reel = new THREE.Group();
    reel.position.set(cx - width / 2 + 0.2, top + 1.4, 0);
    const disc = mesh(new THREE.CylinderGeometry(1.2, 1.2, 0.3, 24), kit.mat("--yellow", { owner }), 0, 0, 0);
    disc.rotation.x = Math.PI / 2;
    reel.add(disc);
    for (let i = 0; i < 3; i++) {
      const spoke = mesh(new THREE.BoxGeometry(2.2, 0.16, 0.34), kit.mat("--yellow-light", { owner }), 0, 0, 0);
      spoke.rotation.z = (i * Math.PI) / 3;
      reel.add(spoke);
    }
    add(reel);
    add(mesh(brickGeometry(P - GAP, P - GAP, 1.2, 0.02), dark, cx - width / 2 + 0.2, top + 0.6, 0));
    const tape = add(mesh(new THREE.PlaneGeometry(2.6, 0.4), kit.trans("--yellow-light", { glow: 0.6, opacity: 0.9 }), cx - width / 2 + 1.5, top + 0.9, 0, false));
    tape.rotation.set(-Math.PI / 2, 0, -0.5);
    extras.reel = reel;
  }
  if (design.kind === "iql") {
    const width = design.cockpit.studs * P;
    const canister = mesh(new THREE.CylinderGeometry(0.7, 0.7, 3 * P, 24).rotateZ(Math.PI / 2), kit.trans("--canopy3d", { opacity: 0.5 }), cx, top + 0.8, 0, false);
    add(canister);
    for (const x of [-1.5 * P, 1.5 * P]) add(mesh(new THREE.CylinderGeometry(0.74, 0.74, 0.3, 24).rotateZ(Math.PI / 2), kit.mat("--yellow", { owner }), cx + x, top + 0.8, 0));
    add(mesh(new THREE.CylinderGeometry(0.4, 0.4, 2.2 * P, 16).rotateZ(Math.PI / 2), kit.trans("--green", { glow: 0.7, opacity: 0.8 }), cx, top + 0.8, 0, false));
    const outer = Math.max(...engines.map((spec) => Math.abs(spec.z) + spec.radius)) + 0.6;
    const from = cx - width / 2, to = Math.max(...engines.map((spec) => spec.front)) - 1;
    for (const side of [-1, 1]) {
      add(rod(steel, new THREE.Vector3(from, -0.35, side * outer), new THREE.Vector3(to, -0.35, side * outer), 0.1));
      for (let x = from; x <= to; x += 3) add(rod(steel, new THREE.Vector3(x, -0.35, side * outer), new THREE.Vector3(x, -0.35, side * (outer - 0.8)), 0.07));
    }
    ["V", "Q", "\u03c0"].forEach((text, i) => {
      const gauge = add(mesh(new THREE.PlaneGeometry(P - 0.06, P - 0.06), kit.raw(new THREE.MeshPhysicalMaterial({ map: letter(text, ["#7fd27f", "#e8e85c", "#e89060"][i]), roughness: 0.2, clearcoat: 0.5, emissive: 0x000000 })), cx - width / 2 - 0.01, top - 0.25, (i - 1) * P, false));
      gauge.rotation.y = -Math.PI / 2;
      extras.gauges.push(gauge);
    });
  }
  build.finish();
  const keep = new Set([...flames, ...binders]);
  extraParts.forEach((part) => part.traverse((child) => keep.add(child)));
  pilot.traverse((child) => keep.add(child));
  bake(group, keep);
  const bounds = new THREE.Box3().setFromObject(group);
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  group.position.set(-center.x, -bounds.min.y, -center.z);
  const outer = new THREE.Group();
  outer.add(group);
  return { body: outer, flames, binders, helmet, pilot, extras, length: size.x, height: size.y, width: size.z };
}

export const DESIGNS = {
  ppo: {
    kind: "ppo",
    binder: "--blue",
    engines: [
      { x: 2.6, z: -2 * P, size: 2, sections: 6 },
      { x: 2.6, z: 2 * P, size: 2, sections: 6, paint: "--blue" },
    ],
    binders: [[0, 1]],
    cockpit: { x: -7.4, studs: 4 },
    cables: [[0, -0.5], [1, 0.5]],
  },
  pqn: {
    kind: "pqn",
    binder: "--orange-light",
    engines: [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5].map((k) => ({ x: 2.2, z: k * P * 1.25, size: 1, sections: 6 })),
    binders: [[0, 5]],
    cockpit: { x: -6.8, studs: 5 },
    cables: [[0, -0.6], [2, -0.2], [3, 0.2], [5, 0.6]],
  },
  dqn: {
    kind: "dqn",
    binder: "--yellow-light",
    engines: [
      { x: 3.4, z: -2 * P, size: 2, sections: 5 },
      { x: 3.4, z: 2 * P, size: 2, sections: 5 },
    ],
    binders: [[0, 1]],
    cockpit: { x: -5.6, studs: 4 },
    cables: [[0, -0.5], [1, 0.5]],
  },
  bc: {
    kind: "bc",
    binder: "--yellow-light",
    engines: [
      { x: 1.6, z: -1.5 * P, size: 2, sections: 4 },
      { x: 1.6, z: 1.5 * P, size: 2, sections: 4 },
    ],
    binders: [[0, 1]],
    cockpit: { x: -5.2, studs: 4 },
    cables: [[0, -0.4], [1, 0.4]],
  },
  iql: {
    kind: "iql",
    binder: "--green",
    engines: [
      { x: 3, z: -2 * P, size: 2, sections: 6 },
      { x: 3, z: 2 * P, size: 2, sections: 6 },
    ],
    binders: [[0, 1]],
    cockpit: { x: -7.2, studs: 4 },
    cables: [[0, -0.4], [1, 0.4]],
  },
};
export const DESIGN_ORDER = ["ppo", "pqn", "dqn", "bc", "iql"];

export const BUILDS = {
  anakin: { scale: 1.45, legs: 1, torso: 1, width: 2, helmet: "dome" },
  sebulba: { scale: 1.5, legs: 0.66, torso: 1, width: 3, helmet: "cap" },
  quadinaros: { scale: 1.45, legs: 2, torso: 1.33, width: 2, helmet: "crest" },
};

export function buildPilot(kit, tint, owner, shape = BUILDS.anakin) {
  const THREE = T();
  const group = new THREE.Group();
  const build = builder(group);
  const suit = kit.mat(tint, { owner });
  const trim = kit.mat(tint === "--blue" || tint === "--green" ? "--text" : `${tint}-light`, { owner });
  const dark = kit.mat("--shadow3d", { owner });
  const steel = kit.mat("--steel3d", { owner });
  const skin = kit.mat("--skin3d", { owner });
  build.brick({ w: 3 * P - GAP, d: 3 * P - GAP, h: BRICK - GAP, material: steel, top: false });
  build.brick({ y: BRICK, w: 3 * P - GAP, d: 3 * P - GAP, h: PLATE - GAP, material: dark, top: false });
  const base = BRICK + PLATE;
  const ring = [[-1, -1], [1, -1], [-1, 1], [1, 1]];
  ring.forEach(([i, j]) => build.stud(dark, new THREE.Vector3(i * P, base, j * P)));
  const figure = new THREE.Group();
  figure.position.y = base;
  const inner = new THREE.Group();
  inner.scale.setScalar(shape.scale);
  figure.add(inner);
  const fb = builder(inner);
  const legs = BRICK * shape.legs;
  const torso = BRICK * shape.torso;
  const wide = shape.width * P;
  fb.brick({ x: -P / 2, w: P - GAP, d: P - GAP, h: legs - GAP, material: dark, top: false });
  fb.brick({ x: P / 2, w: P - GAP, d: P - GAP, h: legs - GAP, material: dark, top: false });
  fb.brick({ y: legs, w: wide - GAP, d: P - GAP, h: PLATE - GAP, material: trim, top: false });
  fb.brick({ y: legs + PLATE, w: wide - GAP, d: P - GAP, h: torso - GAP, material: suit, top: false });
  fb.brick({ y: legs + PLATE + torso, w: wide - GAP, d: P - GAP, h: PLATE - GAP, material: suit, top: false });
  const chest = mesh(new THREE.PlaneGeometry(wide * 0.6, P * 0.5), trim, 0, legs + PLATE + torso * 0.62, P / 2 + 0.002);
  inner.add(chest);
  const arms = [-1, 1].map((side) => {
    const pivot = new THREE.Group();
    pivot.position.set(side * (wide / 2 + 0.22), legs + torso + PLATE * 1.5, 0);
    pivot.add(mesh(brickGeometry(0.42, 0.5, BRICK - GAP, 0.03), suit, 0, -BRICK / 2, 0));
    pivot.add(mesh(new THREE.CylinderGeometry(0.2, 0.2, PLATE - GAP, 12), skin, 0, -BRICK - PLATE / 2, 0));
    inner.add(pivot);
    return pivot;
  });
  const top = head(kit, skin, suit, 1, shape.helmet, kit.mat("--visor3d", { owner, rough: 0.1 }));
  top.position.y = legs + torso + 2 * PLATE;
  inner.add(top);
  fb.finish();
  const moving = new Set();
  arms.forEach((arm) => arm.traverse((child) => moving.add(child)));
  bake(inner, moving);
  group.add(figure);
  build.finish();
  return { group, figure, arms, head: top, base };
}

export function buildGate(kit, owner, map) {
  const THREE = T();
  const group = new THREE.Group();
  const build = builder(group);
  const steel = kit.mat("--steel3d", { owner });
  const dark = kit.mat("--shadow3d", { owner });
  const tan = kit.mat("--ground3d", { owner });
  const levels = 8;
  for (const side of [-1, 1]) {
    const x = side * 5 * P;
    build.brick({ x, w: 4 * P - GAP, d: 4 * P - GAP, h: PLATE - GAP, material: dark, top: false });
    for (let k = 0; k < levels; k++) build.brick({ x, y: PLATE + k * BRICK, w: 2 * P - GAP, d: 2 * P - GAP, h: BRICK - GAP, material: k % 2 ? tan : steel, top: false });
  }
  const spring = PLATE + levels * BRICK;
  const steps = [4 * P, 3 * P, 2 * P];
  steps.forEach((inner, k) => {
    for (const side of [-1, 1]) {
      const w = 6 * P - inner;
      build.brick({ x: side * (inner + w / 2), y: spring + k * BRICK, w: w - GAP, d: 2 * P - GAP, h: BRICK - GAP, material: k % 2 ? steel : dark });
    }
  });
  const rows = steps.length + 1;
  build.brick({ y: spring + steps.length * BRICK, w: 12 * P - GAP, d: 2 * P - GAP, h: BRICK - GAP, material: tan });
  const bulbs = [];
  const top = spring + rows * BRICK;
  const width = 12 * P;
  const glow = kit.trans("--yellow-light", { glow: 1, opacity: 0.85 });
  for (let i = 0; i < Math.round(width / P); i++) {
    const bulb = mesh(new THREE.CylinderGeometry(P / 2 - 0.02, P / 2 - 0.02, PLATE - GAP, 16), glow, -width / 2 + P / 2 + i * P, top + PLATE / 2, P / 2, false);
    group.add(bulb);
    bulbs.push(bulb);
  }
  const tileWidth = 8 * P, tileHeight = 5 * P;
  const bannerMaterial = new THREE.MeshPhysicalMaterial({ map, roughness: 0.18, clearcoat: 0.6, emissiveMap: map, emissive: 0xffffff, emissiveIntensity: 0.35 });
  kit.raw(bannerMaterial);
  group.add(mesh(brickGeometry(tileWidth + P, PLATE - GAP, tileHeight + P, 0.02), dark, 0, 2.2 + tileHeight / 2, 0));
  const banner = mesh(new THREE.PlaneGeometry(tileWidth, tileHeight), bannerMaterial, 0, 2.2 + tileHeight / 2, PLATE / 2 + 0.003, false);
  group.add(banner);
  group.add(rod(steel, new THREE.Vector3(-3 * P, 2.2 + tileHeight + P / 2, 0), new THREE.Vector3(-3 * P, spring, 0), 0.16));
  group.add(rod(steel, new THREE.Vector3(3 * P, 2.2 + tileHeight + P / 2, 0), new THREE.Vector3(3 * P, spring, 0), 0.16));
  build.finish();
  const strip = mesh(brickGeometry(width, P, PLATE, 0.02), glow, 0, top + 0.01, -P / 2, false);
  strip.visible = false;
  group.add(strip);
  return { group, bulbs, strip, banner, height: top + PLATE };
}

export { rod, mesh, wedge, slope };

export function buildCharacter(kit, id, owner) {
  const THREE = T();
  const group = new THREE.Group();
  const build = builder(group);
  const dark = kit.mat("--shadow3d", { owner });
  const steel = kit.mat("--steel3d", { owner });
  build.brick({ w: 4 * P - GAP, d: 4 * P - GAP, h: BRICK - GAP, material: steel, top: false });
  build.brick({ y: BRICK, w: 4 * P - GAP, d: 4 * P - GAP, h: PLATE - GAP, material: dark, top: false });
  const base = BRICK + PLATE;
  [[-1.5, -1.5], [1.5, -1.5], [-1.5, 1.5], [1.5, 1.5]].forEach(([i, j]) => build.stud(dark, new THREE.Vector3(i * P, base, j * P)));
  const figure = new THREE.Group();
  figure.position.y = base;
  const inner = new THREE.Group();
  figure.add(inner);
  const fb = builder(inner);
  const glass = kit.mat("--visor3d", { owner, rough: 0.08 });
  const faced = (skin, mood, radius = 0.39, height = BRICK - GAP) => {
    const face = faceTexture(skin.color.clone().convertLinearToSRGB().getStyle(), mood);
    kit.onTheme(() => face.draw(skin.color.clone().convertLinearToSRGB().getStyle()));
    return mesh(new THREE.CylinderGeometry(radius, radius, height, 24, 1, false, Math.PI, Math.PI * 2), kit.raw(new THREE.MeshPhysicalMaterial({ map: face.map, roughness: 0.3, clearcoat: 0.3 })));
  };
  const arm = (side, x, y, length, sleeve, hand, fist = 0.2) => {
    const pivot = new THREE.Group();
    pivot.position.set(side * x, y, 0);
    pivot.add(mesh(brickGeometry(0.42, 0.5, length - GAP, 0.03), sleeve, 0, -length / 2, 0));
    pivot.add(mesh(new THREE.CylinderGeometry(fist, fist, PLATE * 1.4 - GAP, 14), hand, 0, -length - PLATE * 0.7, 0));
    inner.add(pivot);
    return pivot;
  };
  let arms = [];
  let scale = 1.45;
  let wobble = false;
  if (id === "sebulba") {
    const skin = kit.mat("--alien3d", { owner });
    const vest = kit.mat("--vest3d", { owner });
    const shorts = kit.mat("--leather3d", { owner });
    const legs = PLATE * 2;
    fb.brick({ x: -P * 0.75, w: P - GAP, d: P - GAP, h: legs - GAP, material: skin, top: false });
    fb.brick({ x: P * 0.75, w: P - GAP, d: P - GAP, h: legs - GAP, material: skin, top: false });
    fb.brick({ y: legs, w: 3 * P - GAP, d: 1.5 * P - GAP, h: PLATE * 2 - GAP, material: shorts, top: false });
    fb.brick({ y: legs + PLATE * 2, w: 3 * P - GAP, d: 1.5 * P - GAP, h: BRICK - GAP, material: vest, top: false });
    const top = legs + PLATE * 2 + BRICK;
    arms = [-1, 1].map((side) => arm(side, 1.5 * P + 0.22, top - 0.1, BRICK * 2.1, skin, skin, 0.32));
    const head = new THREE.Group();
    head.position.y = top;
    head.scale.setScalar(1.2);
    const skull = mesh(new THREE.SphereGeometry(0.5, 24, 14), skin, 0, 0.42, 0);
    skull.scale.set(1.75, 0.72, 1.15);
    head.add(skull);
    const face = faced(skin, "grumpy", 0.5, 0.36);
    face.position.set(0, 0.3, 0.1);
    face.scale.set(1.2, 1, 1);
    head.add(face);
    for (const side of [-1, 1]) {
      const rim = mesh(new THREE.CylinderGeometry(0.32, 0.32, 0.14, 20), kit.mat("--yellow-light", { owner }), side * 0.4, 0.55, 0.52);
      rim.rotation.x = Math.PI / 2;
      const lens = mesh(new THREE.CylinderGeometry(0.24, 0.24, 0.16, 20), glass, side * 0.4, 0.55, 0.55);
      lens.rotation.x = Math.PI / 2;
      head.add(rim, lens);
    }
    inner.add(head);
    scale = 1.55;
  } else if (id === "quadinaros") {
    const suit = kit.mat("--suit3d", { owner });
    const skin = kit.mat("--alienpale3d", { owner });
    const legs = BRICK * 2.4;
    fb.brick({ x: -P * 0.4, w: P * 0.6 - GAP, d: P * 0.6 - GAP, h: legs - GAP, material: suit, top: false });
    fb.brick({ x: P * 0.4, w: P * 0.6 - GAP, d: P * 0.6 - GAP, h: legs - GAP, material: suit, top: false });
    fb.brick({ y: legs, w: 1.6 * P - GAP, d: P - GAP, h: BRICK * 1.4 - GAP, material: suit, top: false });
    fb.brick({ y: legs + BRICK * 1.4, w: 1.6 * P - GAP, d: P - GAP, h: PLATE - GAP, material: kit.mat("--yellow", { owner }), top: false });
    const top = legs + BRICK * 1.4 + PLATE;
    fb.brick({ y: top - PLATE, w: 2.6 * P - GAP, d: P * 1.2 - GAP, h: PLATE - GAP, material: kit.mat("--yellow", { owner }), top: false });
    arms = [-1, 1].map((side) => arm(side, 1.3 * P + 0.2, top - 0.1, BRICK * 1.9, suit, skin, 0.22));
    const neck = mesh(new THREE.CylinderGeometry(0.18, 0.24, 1.3, 12), skin, 0, top + 0.65, 0);
    inner.add(neck);
    const head = mesh(new THREE.SphereGeometry(0.64, 24, 16), skin, 0, top + 1.6, 0);
    head.scale.set(1, 0.9, 1.05);
    inner.add(head);
    for (const side of [-1, 1]) inner.add(mesh(new THREE.SphereGeometry(0.085, 12, 8), kit.mat("--eyes3d", { owner }), side * 0.22, top + 1.64, 0.6));
    inner.add(mesh(new THREE.BoxGeometry(0.28, 0.04, 0.04), dark, 0, top + 1.38, 0.64, false));
    scale = 1.35;
    wobble = true;
  } else {
    const tunic = kit.mat("--tunic3d", { owner });
    const leather = kit.mat("--leather3d", { owner });
    const skin = kit.mat("--skinlight3d", { owner });
    const red = kit.mat("--red", { owner });
    const legs = BRICK * 1.2;
    fb.brick({ x: -P / 2, w: P - GAP, d: P - GAP, h: PLATE - GAP, material: leather, top: false });
    fb.brick({ x: P / 2, w: P - GAP, d: P - GAP, h: PLATE - GAP, material: leather, top: false });
    fb.brick({ x: -P / 2, y: PLATE, w: P - GAP, d: P - GAP, h: legs - PLATE - GAP, material: tunic, top: false });
    fb.brick({ x: P / 2, y: PLATE, w: P - GAP, d: P - GAP, h: legs - PLATE - GAP, material: tunic, top: false });
    fb.brick({ y: legs, w: 2 * P - GAP, d: P - GAP, h: PLATE - GAP, material: leather, top: false });
    fb.brick({ y: legs + PLATE, w: 2 * P - GAP, d: P - GAP, h: BRICK - GAP, material: tunic, top: false });
    inner.add(mesh(new THREE.PlaneGeometry(0.22, BRICK * 0.9), red, -0.25, legs + PLATE + BRICK / 2, P / 2 + 0.003));
    const top = legs + PLATE + BRICK;
    arms = [-1, 1].map((side) => arm(side, P + 0.22, top - 0.1, BRICK, tunic, skin));
    const face = faced(skin, "determined");
    face.position.y = top + BRICK / 2;
    inner.add(face);
    const cap = mesh(new THREE.SphereGeometry(0.43, 24, 12, 0, Math.PI * 2, 0, Math.PI * 0.5), leather, 0, top + BRICK * 0.66, 0);
    inner.add(cap);
    for (const side of [-1, 1]) inner.add(mesh(brickGeometry(0.12, 0.34, 0.5, 0.02), leather, side * 0.42, top + BRICK * 0.5, 0));
    const strap = mesh(new THREE.CylinderGeometry(0.42, 0.42, 0.1, 24), leather, 0, top + BRICK * 0.72, 0);
    inner.add(strap);
    for (const side of [-1, 1]) {
      const lens = mesh(new THREE.CylinderGeometry(0.13, 0.13, 0.1, 16), glass, side * 0.15, top + BRICK * 0.78, 0.38);
      lens.rotation.x = Math.PI / 2 - 0.4;
      inner.add(lens);
    }
  }
  inner.scale.setScalar(scale);
  fb.finish();
  const moving = new Set();
  arms.forEach((part) => part.traverse((child) => moving.add(child)));
  bake(inner, moving);
  group.add(figure);
  build.finish();
  return { group, figure, arms, base, wobble };
}
