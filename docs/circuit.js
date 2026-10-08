import { P, BRICK, PLATE, builder, brickGeometry } from "./bricks.js";

const T = () => window.THREE;
const GAP = 0.01;
export const WIDTH = 12;

const POINTS = [
  [0, 0, 20], [40, 0, 20], [74, 0, 17], [104, 0, 3], [117, 0, -22],
  [101, 0, -45], [110, 0, -70], [86, 0, -96], [46, 0, -104], [10, 0, -102], [-26, 0, -100],
  [-56, 1.8, -94], [-78, 3.4, -80], [-92, 1.2, -62], [-104, 0, -40], [-112, 0, -14], [-102, 0, 6], [-84, 0, 17], [-45, 0, 20],
];
const inCanyon = (point) => point.z < -86 && point.x > -44 && point.x < 70;

function hash(key) {
  let value = 2166136261;
  for (let i = 0; i < key.length; i++) { value ^= key.charCodeAt(i); value = Math.imul(value, 16777619); }
  value ^= value >>> 13; value = Math.imul(value, 0x5bd1e995); value ^= value >>> 15;
  return (value >>> 0) / 4294967296;
}

export function buildCircuit(scene, context) {
  const THREE = T();
  const { common, glow, trans, tile, plate, studs } = context;
  const curve = new THREE.CatmullRomCurve3(POINTS.map(([x, y, z]) => new THREE.Vector3(x, y, z)), true, "centripetal");
  const length = curve.getLength();
  const count = Math.ceil(length / 1.5);
  const step = length / count;
  const samples = [];
  for (let i = 0; i < count; i++) {
    const u = i / count;
    const position = curve.getPointAt(u);
    const tangent = curve.getTangentAt(u).normalize();
    samples.push({ u, position, tangent });
  }
  const up = new THREE.Vector3(0, 1, 0);
  samples.forEach((sample, i) => {
    const next = samples[(i + 1) % count].tangent, previous = samples[(i - 1 + count) % count].tangent;
    const flatA = new THREE.Vector3(previous.x, 0, previous.z).normalize(), flatB = new THREE.Vector3(next.x, 0, next.z).normalize();
    sample.turn = new THREE.Vector3().crossVectors(flatA, flatB).y / (2 * step);
  });
  for (let pass = 0; pass < 6; pass++) {
    const smooth = samples.map((sample, i) => (samples[(i - 2 + count) % count].turn + samples[(i - 1 + count) % count].turn + sample.turn + samples[(i + 1) % count].turn + samples[(i + 2) % count].turn) / 5);
    samples.forEach((sample, i) => { sample.turn = smooth[i]; });
  }
  samples.forEach((sample) => {
    sample.bank = Math.max(-0.24, Math.min(0.24, -sample.turn * 7));
    const flat = new THREE.Vector3(sample.tangent.x, 0, sample.tangent.z).normalize();
    const right = new THREE.Vector3().crossVectors(flat, up).normalize();
    sample.right = right.clone().applyAxisAngle(sample.tangent, sample.bank);
    sample.normal = new THREE.Vector3().crossVectors(sample.right, sample.tangent).normalize();
    sample.heading = Math.atan2(-sample.tangent.z, sample.tangent.x);
    sample.lift = (WIDTH / 2 + 1.4) * Math.abs(Math.sin(sample.bank));
  });
  const reachLift = Math.ceil(12 / step);
  const peak = samples.map((sample, i) => {
    let best = 0;
    for (let k = -reachLift; k <= reachLift; k++) best = Math.max(best, samples[(i + k + count) % count].lift);
    return best;
  });
  const eased = peak.map((_, i) => {
    let sum = 0;
    for (let k = -reachLift; k <= reachLift; k++) sum += peak[(i + k + count) % count];
    return sum / (2 * reachLift + 1);
  });
  samples.forEach((sample, i) => { sample.position.y += PLATE + eased[i]; });
  samples.forEach((sample, i) => {
    const next = samples[(i + 1) % count].position, previous = samples[(i - 1 + count) % count].position;
    sample.tangent.subVectors(next, previous).normalize();
    sample.pitch = Math.asin(Math.max(-1, Math.min(1, sample.tangent.y)));
    sample.heading = Math.atan2(-sample.tangent.z, sample.tangent.x);
    const flat = new THREE.Vector3(sample.tangent.x, 0, sample.tangent.z).normalize();
    const right = new THREE.Vector3().crossVectors(flat, up).normalize();
    sample.right = right.applyAxisAngle(sample.tangent, sample.bank);
    sample.normal = new THREE.Vector3().crossVectors(sample.right, sample.tangent).normalize();
  });

  const at = (distance) => {
    const s = ((distance % length) + length) % length;
    const f = s / step;
    const i = Math.floor(f) % count, j = (i + 1) % count, t = f - Math.floor(f);
    const a = samples[i], b = samples[j];
    return {
      position: a.position.clone().lerp(b.position, t),
      tangent: a.tangent.clone().lerp(b.tangent, t).normalize(),
      right: a.right.clone().lerp(b.right, t).normalize(),
      normal: a.normal.clone().lerp(b.normal, t).normalize(),
      heading: a.heading + angle(a.heading, b.heading) * t,
      pitch: a.pitch + (b.pitch - a.pitch) * t,
      bank: a.bank + (b.bank - a.bank) * t,
      turn: a.turn + (b.turn - a.turn) * t,
      u: s / length,
    };
  };
  function angle(from, to) {
    let delta = (to - from) % (Math.PI * 2);
    if (delta > Math.PI) delta -= Math.PI * 2;
    if (delta < -Math.PI) delta += Math.PI * 2;
    return delta;
  }

  const seams = document.createElement("canvas");
  seams.width = seams.height = 128;
  const paint = seams.getContext("2d");
  paint.fillStyle = "#ffffff";
  paint.fillRect(0, 0, 128, 128);
  paint.fillStyle = "rgba(0, 0, 0, 0.35)";
  paint.fillRect(0, 0, 128, 2);
  paint.fillRect(0, 0, 2, 128);
  paint.fillRect(64, 0, 1, 128);
  paint.fillStyle = "rgba(255, 255, 255, 0.45)";
  paint.fillRect(0, 2, 128, 1);
  paint.fillRect(2, 0, 1, 128);
  const seamMap = new THREE.CanvasTexture(seams);
  seamMap.wrapS = seamMap.wrapT = THREE.RepeatWrapping;
  seamMap.anisotropy = 8;
  const road = context.material(new THREE.MeshPhysicalMaterial({ map: seamMap, roughness: 0.6, clearcoat: 0, envMapIntensity: 0.15 }), "--road3d");
  road.userData.env = 0.12;
  const positions = [], uvs = [], indices = [];
  const tileSize = 4 * P;
  for (let i = 0; i <= count; i++) {
    const sample = samples[i % count];
    for (const side of [-1, 1]) {
      const point = sample.position.clone().addScaledVector(sample.right, (side * WIDTH) / 2);
      positions.push(point.x, point.y, point.z);
      uvs.push((side * WIDTH) / 2 / tileSize, (i * step) / tileSize);
    }
    if (i < count) {
      const a = i * 2, b = a + 1, c = a + 2, d = a + 3;
      indices.push(a, b, c, b, d, c);
    }
  }
  const ribbon = new THREE.BufferGeometry();
  ribbon.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  ribbon.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  ribbon.setIndex(indices);
  ribbon.computeVertexNormals();
  road.side = THREE.DoubleSide;
  const surface = new THREE.Mesh(ribbon, road);
  surface.receiveShadow = true;
  scene.add(surface);
  const skirtPositions = [];
  const skirtIndices = [];
  for (let i = 0; i <= count; i++) {
    const sample = samples[i % count];
    for (const side of [-1, 1]) {
      const point = sample.position.clone().addScaledVector(sample.right, (side * WIDTH) / 2);
      skirtPositions.push(point.x, point.y, point.z, point.x, 0, point.z);
    }
    if (i < count) {
      const base = i * 4;
      skirtIndices.push(base, base + 1, base + 4, base + 1, base + 5, base + 4);
      skirtIndices.push(base + 2, base + 6, base + 3, base + 3, base + 6, base + 7);
    }
  }
  const skirt = new THREE.BufferGeometry();
  skirt.setAttribute("position", new THREE.Float32BufferAttribute(skirtPositions, 3));
  skirt.setIndex(skirtIndices);
  skirt.computeVertexNormals();
  const skirtMesh = new THREE.Mesh(skirt, common("--shadow3d"));
  skirtMesh.material.side = THREE.DoubleSide;
  scene.add(skirtMesh);

  const dummy = new THREE.Object3D();
  const basis = new THREE.Matrix4();
  const orient = (sample, point) => {
    const binormal = new THREE.Vector3().crossVectors(sample.tangent, sample.normal).normalize();
    basis.makeBasis(sample.tangent, sample.normal, binormal);
    dummy.quaternion.setFromRotationMatrix(basis);
    dummy.position.copy(point);
    dummy.scale.set(1, 1, 1);
    dummy.updateMatrix();
    return dummy.matrix;
  };

  const dashes = new THREE.InstancedMesh(brickGeometry(4 * P - GAP, P - GAP, 0.05, 0.015), common("--text", { rough: 0.2 }), Math.ceil(length / 7) + 2);
  let dash = 0;
  for (let s = 0; s < length - 4; s += 7) {
    const sample = samples[Math.floor(s / step) % count];
    dashes.setMatrixAt(dash++, orient(sample, sample.position.clone().addScaledVector(sample.normal, 0.03)));
  }
  dashes.count = dash;
  scene.add(dashes);

  const kerbList = [];
  for (let s = 0; s < length; s += 2 * P) {
    const sample = samples[Math.floor(s / step) % count];
    if (Math.abs(sample.turn) < 0.012) continue;
    for (const side of [-1, 1]) kerbList.push([sample, side, kerbList.length]);
  }
  const kerbs = new THREE.InstancedMesh(brickGeometry(2 * P - GAP, P * 1.2 - GAP, PLATE * 1.5, 0.02), new THREE.MeshPhysicalMaterial({ color: 0xffffff, roughness: 0.25, clearcoat: 0.3, envMapIntensity: 0.7 }), Math.max(1, kerbList.length));
  const kerbTones = [];
  kerbList.forEach(([sample, side], i) => {
    kerbs.setMatrixAt(i, orient(sample, sample.position.clone().addScaledVector(sample.right, side * (WIDTH / 2 + P * 0.6))));
    kerbTones.push(Math.floor(i / 2) % 2 ? "--text" : "--red");
  });
  kerbs.count = kerbList.length;
  kerbs.castShadow = kerbs.receiveShadow = true;
  kerbs.userData.tag = "kerbs";
  scene.add(kerbs);

  const support = [];
  for (let s = 0; s < length; s += 3 * P) {
    const sample = samples[Math.floor(s / step) % count];
    const height = sample.position.y - PLATE;
    if (height < BRICK * 0.8) continue;
    for (const side of [-1, 1]) {
      const reach = WIDTH / 2 - 1.2;
      const surface = sample.position.y + sample.right.y * side * reach - Math.abs(sample.right.y) * 1.2;
      const levels = Math.floor((surface - 0.08) / BRICK);
      for (let k = 0; k < levels; k++) support.push([sample.position.x + sample.right.x * side * reach, k * BRICK, sample.position.z + sample.right.z * side * reach, sample.heading, k]);
    }
  }
  const pillars = new THREE.InstancedMesh(brickGeometry(2 * P - GAP, 2 * P - GAP, BRICK - GAP, 0.03), common("--steel3d"), Math.max(1, support.length));
  support.forEach(([x, y, z, heading], i) => {
    dummy.position.set(x, y + BRICK / 2, z);
    dummy.rotation.set(0, heading, 0);
    dummy.quaternion.setFromEuler(dummy.rotation);
    dummy.updateMatrix();
    pillars.setMatrixAt(i, dummy.matrix);
  });
  pillars.count = support.length;
  pillars.castShadow = true;
  pillars.userData.tag = "pillars";
  scene.add(pillars);

  const cliffTop = common("--ridge3d", { rough: 0.45, map: studs.world });
  const cliffSide = common("--ridge3d", { rough: 0.45 });
  const cliffs = [];
  const canyonSamples = samples.filter((sample) => inCanyon(sample.position));
  canyonSamples.forEach((sample, index) => {
    if (index % 2) return;
    const edge = Math.min(1, Math.min(index, canyonSamples.length - index) / 12);
    for (const side of [-1, 1]) {
      for (let row = 0; row < 2; row++) {
        const offset = WIDTH / 2 + 4 + row * 3.2;
        const tall = Math.max(1, Math.round(edge * (6 + 10 * hash(`cliff-${index}-${side}-${row}`) + row * 4)));
        const flat = Math.hypot(sample.right.x, sample.right.z);
        cliffs.push([sample.position.x + (sample.right.x / flat) * side * offset, sample.position.z + (sample.right.z / flat) * side * offset, tall, sample.heading]);
      }
    }
  });
  const cliffCount = cliffs.reduce((sum, [, , tall]) => sum + tall, 0);
  const cliffMesh = new THREE.InstancedMesh(brickGeometry(3.2 - GAP, 3.2 - GAP, BRICK - GAP, 0.04), [cliffTop, cliffSide], Math.max(1, cliffCount));
  let cliffIndex = 0;
  cliffs.forEach(([x, z, tall, heading]) => {
    for (let k = 0; k < tall; k++) {
      dummy.position.set(x, k * BRICK + BRICK / 2, z);
      dummy.rotation.set(0, heading, 0);
      dummy.quaternion.setFromEuler(dummy.rotation);
      dummy.updateMatrix();
      cliffMesh.setMatrixAt(cliffIndex++, dummy.matrix);
    }
  });
  cliffMesh.count = cliffIndex;
  cliffMesh.castShadow = cliffMesh.receiveShadow = true;
  cliffMesh.userData.tag = "cliffs";
  scene.add(cliffMesh);

  const arch = new THREE.Group();
  const archBuild = builder(arch);
  const rock = common("--ridge3d", { rough: 0.45 });
  const rockDark = common("--orange", { rough: 0.45 });
  const span = WIDTH / 2 + 3.4;
  for (const side of [-1, 1]) {
    for (let k = 0; k < 14; k++) archBuild.brick({ x: side * (span + (k > 10 ? -(k - 10) * P : 0)), y: k * BRICK, w: 4 * P - GAP, d: 4 * P - GAP, h: BRICK - GAP, material: k % 3 === 0 ? rockDark : rock, top: false });
  }
  for (let k = 0; k < 3; k++) archBuild.brick({ y: (14 + k) * BRICK, w: 2 * (span + 2 * P) - k * 4 * P - GAP, d: 4 * P - GAP, h: BRICK - GAP, material: k % 2 ? rockDark : rock });
  archBuild.finish();
  const archSample = canyonSamples[Math.floor(canyonSamples.length / 2)] || samples[0];
  arch.position.set(archSample.position.x, 0, archSample.position.z);
  arch.rotation.y = archSample.heading + Math.PI / 2;
  arch.userData.tag = "arch";
  scene.add(arch);

  const finishSample = at(0);
  const checker = document.createElement("canvas");
  checker.width = 16;
  checker.height = 96;
  const checkerPaint = checker.getContext("2d");
  for (let c = 0; c < 2; c++) for (let r = 0; r < 12; r++) {
    checkerPaint.fillStyle = (c + r) % 2 ? "#151515" : "#f2f2f2";
    checkerPaint.fillRect(c * 8, r * 8, 8, 8);
  }
  const checkerMap = new THREE.CanvasTexture(checker);
  checkerMap.magFilter = THREE.NearestFilter;
  const finish = new THREE.Mesh(new THREE.PlaneGeometry(2 * P * 1.25, WIDTH), tile(checkerMap));
  finish.rotation.x = -Math.PI / 2;
  finish.receiveShadow = true;
  const finishHolder = new THREE.Group();
  finishHolder.add(finish);
  finishHolder.position.copy(finishSample.position).add(new THREE.Vector3(0, 0.03, 0));
  finishHolder.rotation.y = finishSample.heading;
  scene.add(finishHolder);

  const gantry = new THREE.Group();
  const gantryBuild = builder(gantry);
  for (const side of [-1, 1]) for (let k = 0; k < 11; k++) gantryBuild.brick({ z: side * (WIDTH / 2 + 2 * P), y: k * BRICK, w: 2 * P - GAP, d: 2 * P - GAP, h: BRICK - GAP, material: k % 2 ? common("--steel3d") : common("--shadow3d"), top: false });
  gantryBuild.brick({ y: 11 * BRICK, w: 3 * P - GAP, d: WIDTH + 6 * P - GAP, h: 2 * BRICK - GAP, material: common("--shadow3d") });
  gantryBuild.finish();
  const sign = new THREE.Mesh(new THREE.PlaneGeometry(WIDTH - 2, BRICK * 0.9), tile(plate("BOONTA EVE", "--red", 256, 32)));
  sign.position.set(-1.5 * P - 0.02, 11.4 * BRICK, 0);
  sign.rotation.y = -Math.PI / 2;
  gantry.add(sign);
  const lamps = [];
  const red = trans("--red", { glow: 1, opacity: 0.9 });
  const green = trans("--green", { glow: 1, opacity: 0.9 });
  const off = common("--shadow3d", { rough: 0.2 });
  for (let i = 0; i < 4; i++) {
    const lamp = new THREE.Mesh(new THREE.CylinderGeometry(P - 0.02, P - 0.02, PLATE, 20), off);
    lamp.rotation.z = Math.PI / 2;
    lamp.position.set(-1.5 * P - PLATE / 2, 12 * BRICK + BRICK * 0.5, (i - 1.5) * 2.2 * P);
    gantry.add(lamp);
    lamps.push(lamp);
  }
  gantry.position.copy(finishSample.position).setY(PLATE);
  gantry.rotation.y = finishSample.heading;
  gantry.userData.tag = "gantry";
  scene.add(gantry);
  const setLights = (lit, go) => {
    lamps.forEach((lamp, i) => { lamp.material = go ? green : i < lit ? red : off; });
  };

  const flags = [];
  const flagTints = ["--red", "--orange", "--yellow", "--blue", "--green"];
  [-80, -66, -52, -38, 44, 58, 72, 86].forEach((s, i) => {
    const sample = at(((s % length) + length) % length);
    const base = sample.position.clone().addScaledVector(sample.right, WIDTH / 2 + 3);
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 9, 8), common("--steel3d", { rough: 0.2 }));
    pole.position.set(base.x, 4.5, base.z);
    pole.castShadow = true;
    pole.userData.tag = "flag";
    scene.add(pole);
    const flag = new THREE.Mesh(brickGeometry(4 * P, 0.08, 3 * P, 0.01), common(flagTints[i % flagTints.length]));
    flag.geometry = flag.geometry.clone();
    flag.geometry.translate(2 * P, 0, 0);
    const holder = new THREE.Group();
    holder.position.set(base.x, 8.2, base.z);
    holder.add(flag);
    holder.rotation.y = sample.heading + Math.PI;
    flag.castShadow = true;
    holder.userData.tag = "flag";
    scene.add(holder);
    flags.push({ holder, phase: i * 0.7, base: holder.rotation.y });
  });

  const cliffAt = (x, z) => {
    let best = 0;
    cliffs.forEach(([cx, cz, tall]) => { if (Math.abs(cx - x) < 2.4 && Math.abs(cz - z) < 2.4) best = Math.max(best, tall * BRICK); });
    return best;
  };
  const footprint = [];
  for (let s = 0; s < length; s += 4) footprint.push(at(s).position);
  const near = (x, z, radius) => footprint.some((point) => (point.x - x) ** 2 + (point.z - z) ** 2 < radius * radius);

  const rockSpots = [];
  for (let i = 0; i < 140 && rockSpots.length < 46; i++) {
    const s0 = hash(`rock-s-${i}`) * length;
    const spot = at(s0);
    const side = hash(`rock-side-${i}`) < 0.5 ? -1 : 1;
    const offset = WIDTH / 2 + 6 + hash(`rock-o-${i}`) * 14;
    const flat = new THREE.Vector3(spot.right.x, 0, spot.right.z).normalize();
    const x = spot.position.x + flat.x * side * offset, z = spot.position.z + flat.z * side * offset;
    if (near(x, z, WIDTH / 2 + 5)) continue;
    if (Math.abs(x) < 112 && z > -26 && z < 92) continue;
    if (Math.hypot(x - 58, z + 12) < 14) continue;
    if (inCanyon({ x, z })) continue;
    rockSpots.push([x, z, 1 + Math.floor(hash(`rock-h-${i}`) * 4), hash(`rock-y-${i}`) * Math.PI]);
  }
  const rockCount = rockSpots.reduce((sum, [, , tall]) => sum + tall, 0);
  const rocks = new THREE.InstancedMesh(brickGeometry(2.4 - GAP, 2.4 - GAP, BRICK - GAP, 0.04), [common("--ridge3d", { rough: 0.45, map: studs.world }), common("--ridge3d", { rough: 0.45 })], Math.max(1, rockCount));
  let rockIndex = 0;
  rockSpots.forEach(([x, z, tall, yaw]) => {
    for (let k = 0; k < tall; k++) {
      const shrink = 1 - k * 0.18;
      dummy.position.set(x, k * BRICK + BRICK / 2, z);
      dummy.rotation.set(0, yaw + k * 0.3, 0);
      dummy.scale.set(shrink * (1.4 + (k === 0 ? 0.6 : 0)), 1, shrink * (1.2 + (k === 0 ? 0.6 : 0)));
      dummy.updateMatrix();
      rocks.setMatrixAt(rockIndex++, dummy.matrix);
    }
  });
  dummy.scale.set(1, 1, 1);
  rocks.count = rockIndex;
  rocks.castShadow = rocks.receiveShadow = true;
  rocks.userData.tag = "rocks";
  scene.add(rocks);

  return {
    length,
    at,
    near,
    setLights,
    cliffAt,
    arch,
    flags,
    kerbs,
    kerbTones,
    samples,
    step,
  };
}
