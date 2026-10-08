import { SCREEN, drawBanner } from "./screen.js";
import { P, BRICK, PLATE, STUD, builder, brickGeometry, studGeometry, studMaps } from "./bricks.js";
import { buildPod, buildGate, DESIGNS, DESIGN_ORDER, mesh } from "./models.js";
import { buildMinifigure, pose } from "./figures.js";
import { buildCircuit, WIDTH } from "./circuit.js";

const GZ = 57;
const POD_TINTS = ["--red", "--orange", "--yellow", "--blue", "--green"];
const CHARACTER_TINTS = { anakin: "--red", sebulba: "--orange", quadinaros: "--yellow" };
const GRID = -16;
const DYNO = { x: 19, y: 2.4, z: 31 };
const REEL = { x: 5, y: 4.2, z: 31 };
const LEARNER = { x: 58, z: -12 };
const ROW_Z = { track: GZ, pilot: GZ - 1, podracer: GZ };
const ROW_Y = { track: 8.5, pilot: 5.2, podracer: 11 };
const BAYS = {
  podracer: { x: -88, z: GZ, along: [0, 0, -1], face: [1, 0, 0], yaw: Math.PI / 2, rise: 0.14 },
  pilot: { x: 0, z: GZ - 1, along: [1, 0, 0], face: [0, 0, 1], yaw: 0, rise: 0.2 },
  track: { x: 90, z: GZ, along: [0, 0, 1], face: [-1, 0, 0], yaw: -Math.PI / 2, rise: 0.18 },
};
const bayPoint = (kind, offset, y = 0) => {
  const bay = BAYS[kind];
  return [bay.x + bay.along[0] * offset, y, bay.z + bay.along[2] * offset];
};
const STAND_TOP = 2 * BRICK + PLATE;
const GAP = 0.01;

function hash(key) {
  let value = 2166136261;
  for (let i = 0; i < key.length; i++) { value ^= key.charCodeAt(i); value = Math.imul(value, 16777619); }
  value ^= value >>> 13; value = Math.imul(value, 0x5bd1e995); value ^= value >>> 15;
  return (value >>> 0) / 4294967296;
}
function spread(count, spacing) {
  return Array.from({ length: count }, (_, i) => (i - (count - 1) / 2) * spacing);
}
function speedFor(sps) {
  if (!(sps > 0)) return 0;
  const level = Math.max(0, Math.min(1, (Math.log10(sps) - 2.5) / 2.5));
  return 10 + 62 * level;
}
function angleTo(from, to) {
  let delta = (to - from) % (Math.PI * 2);
  if (delta > Math.PI) delta -= Math.PI * 2;
  if (delta < -Math.PI) delta += Math.PI * 2;
  return delta;
}
const snapTo = (value) => Math.round(value / P) * P;

export function createWorld(canvas, options) {
  const THREE = window.THREE;
  if (!THREE) return null;
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: (window.devicePixelRatio || 1) < 1.5, powerPreference: "high-performance" });
  } catch (error) {
    return null;
  }
  if (!renderer.getContext()) return null;
  const coarse = window.matchMedia("(pointer: coarse)").matches;
  const TIERS = [{ ratio: 1, shadow: 512, soft: false, studs: false }, { ratio: 1.5, shadow: 1024, soft: false, studs: false }, { ratio: 2, shadow: 2048, soft: true, studs: true }];
  const asked = new URLSearchParams(location.search).get("quality");
  const pinned = Boolean(asked) && TIERS[Number(asked)] !== undefined;
  let tier = pinned ? Number(asked) : coarse || (navigator.hardwareConcurrency || 8) <= 4 ? 1 : 2;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = TIERS[tier].soft ? THREE.PCFSoftShadowMap : THREE.PCFShadowMap;
  renderer.physicallyCorrectLights = false;
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  if (!THREE.CanvasTexture.srgb) {
    const Base = THREE.CanvasTexture;
    const Encoded = class extends Base {
      constructor(...args) {
        super(...args);
        this.encoding = THREE.sRGBEncoding;
      }
    };
    Encoded.srgb = true;
    THREE.CanvasTexture = Encoded;
  }
  const { menu, screen, reduced } = options;
  const studs = studMaps(renderer);

  const tokens = {};
  let css = getComputedStyle(document.documentElement);
  const cssColor = (name) => css.getPropertyValue(name).trim() || "#888";
  const tone = (name) => {
    if (!tokens[name]) tokens[name] = new THREE.Color();
    tokens[name].setStyle(cssColor(name)).convertSRGBToLinear();
    return tokens[name];
  };
  const hsl = {};
  const neutral = new Set(["--ground3d", "--ridge3d", "--slab3d", "--road3d", "--steel3d", "--shadow3d", "--skin3d", "--tunic3d", "--leather3d", "--skinlight3d", "--alien3d", "--vest3d", "--suit3d", "--alienpale3d"]);
  const toy = (color, token) => {
    if (neutral.has(token)) return color;
    color.convertLinearToSRGB().getHSL(hsl);
    if (hsl.s > 0.18) color.setHSL(hsl.h, Math.min(0.8, hsl.s * 1.2 + 0.05), Math.max(0.32, Math.min(0.62, hsl.l * 1.05)));
    return color.convertSRGBToLinear();
  };
  const paints = [];
  const themed = [];
  const kit = {
    mat(token, { owner = null, rough = 0.28, map = null } = {}) {
      const made = new THREE.MeshPhysicalMaterial({ color: toy(tone(token).clone(), token), roughness: rough, metalness: 0, clearcoat: 0.6, clearcoatRoughness: 0.08, envMapIntensity: 0.75 });
      if (map) { made.map = map.map; made.bumpMap = map.bumpMap; made.bumpScale = 0.025; }
      paints.push({ material: made, token, owner, toy: true });
      if (owner) owner.materials.push(made);
      return made;
    },
    trans(token, { owner = null, glow = 0, opacity = 0.6 } = {}) {
      const made = new THREE.MeshPhysicalMaterial({ color: toy(tone(token).clone(), token), roughness: 0.06, metalness: 0, transparent: true, opacity, depthWrite: false, side: THREE.DoubleSide, envMapIntensity: 1.2 });
      paints.push({ material: made, token, owner, toy: true, glow });
      return made;
    },
    raw(material) { return material; },
    onTheme(callback) { themed.push(callback); },
  };
  const shared = {};
  const common = (token, options = {}) => {
    const key = `${token}:${options.rough || ""}:${options.map ? "s" : ""}`;
    if (!shared[key]) shared[key] = kit.mat(token, options);
    return shared[key];
  };
  const glow = (token) => {
    const key = `glow:${token}`;
    if (!shared[key]) {
      shared[key] = new THREE.MeshBasicMaterial({ color: tone(token).clone() });
      paints.push({ material: shared[key], token });
    }
    return shared[key];
  };
  const textures = [];
  const plate = (text, token, width = 192, height = 32, bright = false) => {
    const source = document.createElement("canvas");
    source.width = width;
    source.height = height;
    const map = new THREE.CanvasTexture(source);
    map.anisotropy = 4;
    const draw = () => {
      const context = source.getContext("2d");
      context.fillStyle = bright ? cssColor(token) : "rgb(20, 20, 22)";
      context.fillRect(0, 0, width, height);
      if (bright) {
        context.fillStyle = "rgba(255, 255, 255, 0.35)";
        context.fillRect(0, 0, width, Math.max(2, height * 0.08));
      }
      context.fillStyle = bright ? "rgb(250, 248, 240)" : cssColor(token);
      context.font = `${Math.round(height * 0.42)}px "Press Start 2P", monospace`;
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(text, width / 2, height / 2 + 1);
      map.needsUpdate = true;
    };
    draw();
    textures.push(draw);
    return map;
  };
  const tileMaterial = (map) => {
    const made = new THREE.MeshPhysicalMaterial({ map, roughness: 0.18, clearcoat: 0.5, clearcoatRoughness: 0.1, envMapIntensity: 0.6 });
    return made;
  };

  const GLYPHS = {
    B: ["1111.", "1...1", "1...1", "1111.", "1...1", "1...1", "1111."],
    O: [".111.", "1...1", "1...1", "1...1", "1...1", "1...1", ".111."],
    N: ["1...1", "11..1", "1.1.1", "1.1.1", "1..11", "1...1", "1...1"],
    T: ["11111", "..1..", "..1..", "..1..", "..1..", "..1..", "..1.."],
    A: [".111.", "1...1", "1...1", "11111", "1...1", "1...1", "1...1"],
  };
  const wordmarks = [];
  const wordmark = (pixel = 1.2) => {
    const group = new THREE.Group();
    group.userData.wordmark = true;
    wordmarks.push(group);
    const bands = ["--red", "--red", "--orange", "--orange", "--orange", "--yellow", "--yellow"];
    const geometry = brickGeometry(pixel - 0.03, pixel * 0.9, pixel - 0.03, 0.05);
    const studGeometryLocal = new THREE.CylinderGeometry(pixel * 0.3, pixel * 0.3, pixel * 0.2, 12);
    studGeometryLocal.translate(0, pixel * 0.1, 0);
    const cells = new Map();
    const word = "BOONTA";
    const columns = word.length * 6 - 1;
    [...word].forEach((letter, index) => GLYPHS[letter].forEach((line, row) => [...line].forEach((cell, column) => {
      if (cell !== "1") return;
      const token = bands[row];
      if (!cells.has(token)) cells.set(token, { boxes: [], studs: [] });
      const x = (index * 6 + column - (columns - 1) / 2) * pixel;
      const y = (6 - row) * pixel + pixel / 2;
      cells.get(token).boxes.push([x, y]);
      const above = row > 0 && GLYPHS[letter][row - 1][column] === "1";
      if (!above) cells.get(token).studs.push([x, y + pixel / 2]);
    })));
    const matrix = new THREE.Object3D();
    cells.forEach(({ boxes, studs }, token) => {
      const material = common(token, { rough: 0.25 });
      const blocks = new THREE.InstancedMesh(geometry, material, boxes.length);
      boxes.forEach(([x, y], i) => { matrix.position.set(x, y, 0); matrix.updateMatrix(); blocks.setMatrixAt(i, matrix.matrix); });
      blocks.castShadow = true;
      const knobs = new THREE.InstancedMesh(studGeometryLocal, material, studs.length);
      studs.forEach(([x, y], i) => { matrix.position.set(x, y, 0); matrix.updateMatrix(); knobs.setMatrixAt(i, matrix.matrix); });
      knobs.userData.studs = true;
      group.add(blocks, knobs);
    });
    const backing = new THREE.Mesh(brickGeometry((columns + 2) * pixel, pixel * 0.5, 8 * pixel, 0.05), common("--shadow3d"));
    backing.position.set(0, 3.5 * pixel + pixel / 2, -pixel * 0.65);
    backing.castShadow = true;
    group.add(backing);
    group.userData.width = (columns + 2) * pixel;
    group.userData.height = 8 * pixel;
    return group;
  };

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x000000, 220, 950);

  const pmrem = new THREE.PMREMGenerator(renderer);
  const studio = new THREE.Scene();
  const domeGeometry = new THREE.SphereGeometry(60, 24, 12);
  const domeColors = [];
  const domePositions = domeGeometry.attributes.position;
  for (let i = 0; i < domePositions.count; i++) {
    const y = domePositions.getY(i) / 60;
    const shade = y > 0 ? 0.55 + 0.45 * y : 0.18 + 0.2 * (1 + y);
    domeColors.push(shade, shade * 0.98, shade * 0.95);
  }
  domeGeometry.setAttribute("color", new THREE.Float32BufferAttribute(domeColors, 3));
  studio.add(new THREE.Mesh(domeGeometry, new THREE.MeshBasicMaterial({ side: THREE.BackSide, vertexColors: true })));
  [[-30, 40, 30, 3.5], [35, 30, -10, 2.2], [0, 55, -30, 2.8]].forEach(([x, y, z, power]) => {
    const panel = new THREE.Mesh(new THREE.PlaneGeometry(26, 14), new THREE.MeshBasicMaterial({ color: new THREE.Color(power, power * 0.97, power * 0.92), side: THREE.DoubleSide }));
    panel.position.set(x, y, z);
    panel.lookAt(0, 0, 0);
    studio.add(panel);
  });
  scene.environment = pmrem.fromScene(studio, 0.03).texture;

  const skyUniforms = { top: { value: new THREE.Color() }, horizon: { value: new THREE.Color() } };
  scene.add(new THREE.Mesh(
    new THREE.SphereGeometry(1400, 24, 14),
    new THREE.ShaderMaterial({
      uniforms: skyUniforms,
      side: THREE.BackSide,
      depthWrite: false,
      fog: false,
      vertexShader: "varying vec3 vPosition; void main() { vPosition = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
      fragmentShader: "uniform vec3 top; uniform vec3 horizon; varying vec3 vPosition; void main() { float h = clamp(normalize(vPosition).y, 0.0, 1.0); gl_FragColor = linearToOutputTexel(vec4(mix(horizon, top, pow(h, 0.55)), 1.0)); }",
    }),
  ));

  const hemisphere = new THREE.HemisphereLight(0xc8d4ff, 0x3a2a20, 0.5);
  scene.add(hemisphere);
  const key = new THREE.DirectionalLight(0xffe0b8, 1.0);
  key.position.set(-70, 120, GZ + 70);
  key.target.position.set(0, 0, 14);
  key.castShadow = true;
  key.shadow.mapSize.set(TIERS[tier].shadow, TIERS[tier].shadow);
  Object.assign(key.shadow.camera, { left: -80, right: 80, top: 80, bottom: -80, near: 10, far: 400 });
  key.shadow.bias = -0.0004;
  key.shadow.normalBias = 0.03;
  key.shadow.radius = 3;
  scene.add(key, key.target);
  const fill = new THREE.DirectionalLight(0x9db4ff, 0.35);
  fill.position.set(80, 40, -60);
  scene.add(fill);

  const tile = (map) => tileMaterial(map);
  const circuit = buildCircuit(scene, {
    common,
    glow,
    trans: (token, options) => kit.trans(token, options),
    tile,
    plate: (...args) => plate(...args),
    studs,
    material: (made, token) => { paints.push({ material: made, token }); return made; },
  });
  const LAP = circuit.length;
  const place = (distance) => circuit.at(distance);
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(2600, 2600), kit.mat("--ground3d", { rough: 0.45, map: studs.sized(2600 / P, 2600 / P) }));
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  ground.material.roughness = 0.75;
  ground.material.clearcoat = 0;
  ground.material.userData.env = 0.2;
  scene.add(ground);

  const dummy = new THREE.Object3D();
  const terraces = [];
  for (let x = -460; x <= 460; x += 16) {
    for (let z = -440; z <= 460; z += 16) {
      const reach = (x / 140) ** 2 + ((z + 30) / 110) ** 2;
      if (reach > 14 || (Math.abs(x) < 112 && z > 22 && z < 92)) continue;
      if (circuit.near(x, z, 22)) continue;
      if (Math.abs(x) < 52 && z > -26 && z < 22) continue;
      if (Math.hypot(x - LEARNER.x, z - LEARNER.z) < 14) continue;
      const ramp = Math.min(1, Math.max(0.25, (reach - 0.6) / 2.2));
      const dune = 4 + 5 * Math.sin(x * 0.018 + Math.cos(z * 0.011) * 2) * Math.cos(z * 0.021) + 3 * Math.sin((x + z) * 0.043);
      const levels = Math.round((ramp * Math.max(0, dune)) / BRICK);
      if (levels < 1) continue;
      terraces.push([x, z, levels]);
    }
  }
  const sandTop = kit.mat("--ground3d", { rough: 0.45, map: studs.world });
  const sandSide = kit.mat("--ground3d", { rough: 0.45 });
  const duneGeometry = brickGeometry(16 - 0.02, 16 - 0.02, BRICK, 0.06);
  const dunes = new THREE.InstancedMesh(duneGeometry, [sandTop, sandSide], terraces.reduce((sum, [, , levels]) => sum + levels, 0));
  let duneIndex = 0;
  terraces.forEach(([x, z, levels]) => {
    for (let level = 0; level < levels; level++) {
      dummy.position.set(x, BRICK / 2 + level * BRICK, z);
      dummy.updateMatrix();
      dunes.setMatrixAt(duneIndex++, dummy.matrix);
    }
  });
  dunes.receiveShadow = true;
  dunes.userData.tag = "dunes";
  scene.add(dunes);

  const mesaTop = kit.mat("--ridge3d", { rough: 0.5, map: studs.world });
  const mesaSide = kit.mat("--ridge3d", { rough: 0.5 });
  const mesaCount = 40;
  const mesas = new THREE.InstancedMesh(brickGeometry(1, 1, 1, 0.01), [mesaTop, mesaSide], mesaCount * 4);
  let mesaIndex = 0;
  for (let i = 0; i < mesaCount; i++) {
    const angle = (i / mesaCount) * Math.PI * 2 + hash(`m-a-${i}`) * 0.1;
    const distance = 600 + 220 * hash(`m-r-${i}`);
    const width = 50 + 60 * hash(`m-w-${i}`);
    let base = 0;
    for (let tier = 0; tier < 4; tier++) {
      const tall = 10 + 18 * hash(`m-h-${i}-${tier}`);
      const span = width * (1 - tier * 0.2);
      dummy.position.set(Math.cos(angle) * distance, base + tall / 2, Math.sin(angle) * distance);
      dummy.rotation.set(0, hash(`m-y-${i}`) * 3, 0);
      dummy.scale.set(span, tall, span * 0.8);
      dummy.updateMatrix();
      mesas.setMatrixAt(mesaIndex++, dummy.matrix);
      base += tall;
    }
  }
  dummy.scale.set(1, 1, 1);
  dummy.rotation.set(0, 0, 0);
  scene.add(mesas);

  const starCount = 700;
  const starPositions = new Float32Array(starCount * 3);
  for (let i = 0; i < starCount; i++) {
    const theta = hash(`sky-t-${i}`) * Math.PI * 2;
    const phi = 0.15 + hash(`sky-p-${i}`) * 1.3;
    starPositions.set([Math.cos(theta) * Math.cos(phi) * 1200, Math.sin(phi) * 1200, Math.sin(theta) * Math.cos(phi) * 1200], i * 3);
  }
  const starGeometry = new THREE.BufferGeometry();
  starGeometry.setAttribute("position", new THREE.BufferAttribute(starPositions, 3));
  const stars = new THREE.Points(starGeometry, new THREE.PointsMaterial({ size: 1.6, sizeAttenuation: false, color: 0xc8c8d8, fog: false }));
  scene.add(stars);
  const moon = new THREE.Mesh(new THREE.SphereGeometry(42, 24, 16), glow("--moon"));
  moon.material.fog = false;
  moon.position.set(560, 120, -900);
  scene.add(moon);
  const second = new THREE.Mesh(new THREE.SphereGeometry(22, 16, 10), glow("--sun3d"));
  second.material.fog = false;
  second.position.set(430, 95, -960);
  scene.add(second);

  const world = new THREE.Group();
  scene.add(world);
  const build = builder(world);

  const circleStuds = (material, cx, cz, radius, y) => {
    for (let x = -radius; x <= radius; x += P) for (let z = -radius; z <= radius; z += P) {
      const sx = snapTo(x) + P / 2, sz = snapTo(z) + P / 2;
      if (sx * sx + sz * sz < (radius - P * 0.6) ** 2) build.stud(material, new THREE.Vector3(cx + sx, y, cz + sz));
    }
  };
  const stands = new THREE.Group();
  const standZ = 20 - WIDTH / 2 - 5;
  const standBuild = builder(stands);
  const tierWidth = 84;
  for (let tier = 0; tier < 4; tier++) {
    const tall = 3 * BRICK * (tier + 1);
    for (let k = 0; k < 3 * (tier + 1); k++) {
      standBuild.brick({ z: -tier * 6 * P, y: k * BRICK, w: tierWidth, d: 6 * P - GAP, h: BRICK - GAP, material: k % 2 ? common("--steel3d") : common("--shadow3d"), top: k === 3 * (tier + 1) - 1 });
    }
    void tall;
  }
  const crowdCount = 160;
  const bodies = new THREE.InstancedMesh(brickGeometry(2 * P - GAP, P - GAP, BRICK * 1.6, 0.03), new THREE.MeshPhysicalMaterial({ color: 0xffffff, roughness: 0.3, clearcoat: 0.25, envMapIntensity: 0.7 }), crowdCount);
  const heads = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.39, 0.39, BRICK - GAP, 16), common("--skin3d"), crowdCount);
  const shirts = ["--red", "--orange", "--yellow", "--blue", "--text", "--dim", "--green"];
  const seats = [];
  for (let i = 0; i < crowdCount; i++) {
    const tier = i % 4;
    const top = 3 * BRICK * (tier + 1);
    const x = snapTo(-tierWidth / 2 + 3 + (tierWidth - 6) * hash(`fan-${i}`));
    seats.push({ position: new THREE.Vector3(x, top, -tier * 6 * P - P), shirt: shirts[Math.floor(hash(`shirt-${i}`) * shirts.length)] });
  }
  stands.add(bodies, heads);
  bodies.castShadow = heads.castShadow = true;
  const placeCrowd = (clock, cheering) => {
    seats.forEach((seat, i) => {
      const hop = cheering && !reduced ? Math.max(0, Math.sin(clock * 6 + i)) * 0.5 : 0;
      dummy.position.set(seat.position.x, seat.position.y + BRICK * 0.8 + hop, seat.position.z);
      dummy.updateMatrix();
      bodies.setMatrixAt(i, dummy.matrix);
      dummy.position.y += BRICK * 0.8 + BRICK / 2;
      dummy.updateMatrix();
      heads.setMatrixAt(i, dummy.matrix);
    });
    bodies.instanceMatrix.needsUpdate = true;
    heads.instanceMatrix.needsUpdate = true;
  };
  placeCrowd(0, false);
  const screenWidth = 30, screenHeight = (screenWidth * SCREEN.height) / SCREEN.width;
  const screenBase = 15;
  const screenZ = -4 * 6 * P - 2;
  standBuild.brick({ x: 0, y: screenBase - P, z: screenZ, w: screenWidth + 2 * P, d: P - GAP, h: screenHeight + 2 * P, material: common("--shadow3d"), top: true });
  for (const x of [-10, 10]) for (let k = 0; k < Math.floor((screenBase - P) / BRICK); k++) standBuild.brick({ x, y: k * BRICK, z: screenZ, w: 2 * P - GAP, d: 2 * P - GAP, h: BRICK - GAP, material: k % 2 ? common("--steel3d") : common("--shadow3d"), top: false });
  standBuild.finish();
  const screenTexture = new THREE.CanvasTexture(screen);
  screenTexture.magFilter = THREE.NearestFilter;
  screenTexture.minFilter = THREE.NearestFilter;
  screenTexture.generateMipmaps = false;
  const screenPlane = new THREE.Mesh(new THREE.PlaneGeometry(screenWidth, screenHeight), new THREE.MeshBasicMaterial({ map: screenTexture, fog: false, toneMapped: false }));
  screenPlane.position.set(0, screenBase + screenHeight / 2, screenZ + P / 2 + 0.02);
  stands.add(screenPlane);
  const billboard = wordmark(0.9);
  billboard.position.set(-33.5, 12 * BRICK, -3 * 6 * P);
  stands.add(billboard);
  stands.position.set(0, 0, standZ);
  const finishMark = wordmark(0.3);
  const finishSpot = circuit.at(0);
  const finishFace = new THREE.Vector3(Math.cos(finishSpot.heading), 0, -Math.sin(finishSpot.heading));
  finishMark.position.copy(finishSpot.position).setY(12.4 * BRICK + PLATE - 2.3).addScaledVector(finishFace, -(1.5 * P + 0.45));
  finishMark.rotation.y = finishSpot.heading - Math.PI / 2;
  const finishBack = wordmark(0.3);
  finishBack.position.copy(finishSpot.position).setY(12.4 * BRICK + PLATE - 2.3).addScaledVector(finishFace, 1.5 * P + 0.45);
  finishBack.rotation.y = finishSpot.heading + Math.PI / 2;
  finishMark.userData.tag = finishBack.userData.tag = "gantry";
  scene.add(finishMark, finishBack);
  const canyonBoard = new THREE.Group();
  const canyonMark = wordmark(1.4);
  canyonMark.position.y = 8;
  canyonBoard.add(canyonMark);
  for (const x of [-16, 16]) {
    const leg = new THREE.Mesh(brickGeometry(2 * P - GAP, 2 * P - GAP, 8.2, 0.03), common("--steel3d"));
    leg.position.set(x, 4.1, -1);
    leg.castShadow = true;
    canyonBoard.add(leg);
  }
  canyonBoard.position.set(-40, 0, -128);
  for (const side of [-1, 1]) {
    const archMark = wordmark(0.5);
    archMark.position.set(0, 17 * BRICK, side * (2 * P + 0.3));
    archMark.rotation.y = side > 0 ? 0 : Math.PI;
    circuit.arch.add(archMark);
  }
  canyonBoard.userData.tag = "billboard";
  scene.add(canyonBoard);
  stands.userData.tag = "stands";
  scene.add(stands);
  const screenCenter = new THREE.Vector3(0, screenBase + screenHeight / 2, standZ + screenZ);

  const slabMaterial = common("--slab3d", { rough: 0.3 });
  const slabWidth = 94 * P, slabDepth = 50 * P;
  const slab = mesh(brickGeometry(slabWidth, slabDepth, PLATE - GAP, 0.03), slabMaterial, 0, PLATE / 2, GZ + 1);
  slab.receiveShadow = true;
  slab.userData.tag = "slab";
  scene.add(slab);
  const slabStuds = new THREE.InstancedMesh(studGeometry(), slabMaterial, 72 * 50);
  let slabStud = 0;
  for (let i = 0; i < 72; i++) for (let j = 0; j < 50; j++) {
    dummy.position.set(-slabWidth / 2 + P / 2 + i * P, PLATE, GZ + 1 - slabDepth / 2 + P / 2 + j * P);
    dummy.updateMatrix();
    slabStuds.setMatrixAt(slabStud++, dummy.matrix);
  }
  slabStuds.receiveShadow = true;
  scene.add(slabStuds);

  const hangar = new THREE.Group();
  const hangarBuild = builder(hangar);
  const door = new THREE.Group();
  const doorBuild = builder(door);
  for (const side of [-1, 1]) {
    const x = side * (slabWidth / 2 + P);
    for (let k = 0; k < 8; k++) hangarBuild.brick({ x, y: k * BRICK, z: GZ + 1, w: 2 * P - GAP, d: slabDepth - GAP, h: BRICK - GAP, material: k === 7 ? common("--orange") : common("--shadow3d"), top: k === 7 });
    for (let k = 0; k < 15; k++) doorBuild.brick({ x, y: k * BRICK, z: GZ - slabDepth / 2 + 1 + P, w: 3 * P - GAP, d: 3 * P - GAP, h: BRICK - GAP, material: k % 2 ? common("--steel3d") : common("--shadow3d"), top: false });
  }
  doorBuild.brick({ y: 15 * BRICK, z: GZ - slabDepth / 2 + 1 + P, w: slabWidth + 7 * P, d: 3 * P - GAP, h: 2 * BRICK - GAP, material: common("--shadow3d") });
  doorBuild.finish();
  hangarBuild.finish();
  const windowMaterial = kit.trans("--window3d", { glow: 0.5, opacity: 0.85 });
  const rackColors = ["--red", "--yellow", "--blue", "--green", "--orange"];
  for (const side of [-1, 1]) {
    const face = side * (slabWidth / 2) - side * 0.02;
    for (let i = 0; i < 6; i++) {
      const z = GZ + 1 - slabDepth / 2 + 3 + i * (slabDepth - 6) / 5;
      const pane = new THREE.Mesh(new THREE.PlaneGeometry(3 * P, 2 * BRICK), windowMaterial);
      pane.position.set(face, 5 * BRICK, z);
      pane.rotation.y = -side * Math.PI / 2;
      hangar.add(pane);
      const frame = new THREE.Mesh(brickGeometry(0.1, 3.4 * P, 0.2, 0.01), common("--steel3d"));
      frame.position.set(face, 5 * BRICK + BRICK, z);
      hangar.add(frame);
      if (i % 2 === 0) {
        for (let t = 0; t < 4; t++) {
          const tool = new THREE.Mesh(brickGeometry(0.2, P * 0.6, BRICK * (0.8 + 0.3 * (t % 2)), 0.01), common(rackColors[(i + t) % rackColors.length]));
          tool.position.set(face - side * 0.12, 2.2 * BRICK, z + (t - 1.5) * P);
          hangar.add(tool);
        }
        const rack = new THREE.Mesh(brickGeometry(0.2, 4 * P, 0.16, 0.01), common("--steel3d"));
        rack.position.set(face - side * 0.12, 2.9 * BRICK, z);
        hangar.add(rack);
      }
    }
  }
  const hangarSign = wordmark(1.1);
  hangarSign.position.set(0, 17 * BRICK, GZ - slabDepth / 2 + 1 + P);
  door.add(hangarSign);
  const baySign = new THREE.Group();
  const bayMark = wordmark(0.62);
  bayMark.userData.bay = "pilot";
  bayMark.position.set(0, 3.6, 0);
  baySign.add(bayMark);
  for (const x of [-9, 9]) {
    const post = new THREE.Mesh(brickGeometry(P - GAP, P - GAP, 3.6, 0.03), common("--steel3d"));
    post.position.set(x, 1.8 + PLATE, -0.5);
    post.castShadow = true;
    baySign.add(post);
  }
  baySign.position.set(0, 0, GZ - 13);
  door.add(baySign);
  baySign.visible = false;
  door.userData.tag = "door";
  hangar.add(door);
  hangar.userData.tag = "hangar";
  scene.add(hangar);
  const lamps = [-1, 1].map((side) => {
    const lamp = new THREE.PointLight(0xffd8a8, 0.5, 90, 1.2);
    lamp.position.set(side * 26, 24, GZ - 10);
    scene.add(lamp);
    return lamp;
  });

  const blobCanvas = document.createElement("canvas");
  blobCanvas.width = blobCanvas.height = 64;
  const blobPaint = blobCanvas.getContext("2d");
  const blobGradient = blobPaint.createRadialGradient(32, 32, 0, 32, 32, 32);
  blobGradient.addColorStop(0, "rgba(0,0,0,0.55)");
  blobGradient.addColorStop(0.6, "rgba(0,0,0,0.25)");
  blobGradient.addColorStop(1, "rgba(0,0,0,0)");
  blobPaint.fillStyle = blobGradient;
  blobPaint.fillRect(0, 0, 64, 64);
  const blobMaterial = new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(blobCanvas), transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2 });
  const blob = (radius, x = 0, y = 0.02, z = 0) => {
    const made = new THREE.Mesh(new THREE.PlaneGeometry(radius * 2, radius * 2), blobMaterial);
    made.rotation.x = -Math.PI / 2;
    made.position.set(x, y, z);
    made.userData.ring = true;
    made.renderOrder = 1;
    return made;
  };
  const items = [];
  const pickables = [];
  const lockCanvas = document.createElement("canvas");
  lockCanvas.width = lockCanvas.height = 64;
  const lockPaint = lockCanvas.getContext("2d");
  lockPaint.fillStyle = "#2a2a2c";
  lockPaint.fillRect(0, 0, 64, 64);
  lockPaint.strokeStyle = "#d8d8d8";
  lockPaint.lineWidth = 6;
  lockPaint.beginPath(); lockPaint.arc(32, 26, 11, Math.PI, 0); lockPaint.stroke();
  lockPaint.fillStyle = "#d8d8d8";
  lockPaint.fillRect(16, 26, 32, 24);
  lockPaint.fillStyle = "#2a2a2c";
  lockPaint.fillRect(30, 33, 4, 10);
  const lockMaterial = tileMaterial(new THREE.CanvasTexture(lockCanvas));
  const lockGeometry = brickGeometry(2 * P - GAP, 2 * P - GAP, PLATE, 0.03);
  const register = (item) => {
    const lock = new THREE.Group();
    lock.add(new THREE.Mesh(lockGeometry, common("--shadow3d")));
    const face = new THREE.Mesh(new THREE.PlaneGeometry(2 * P - 0.1, 2 * P - 0.1), lockMaterial);
    face.position.z = PLATE / 2 + 0.005;
    lock.add(face);
    lock.rotation.x = -0.35;
    lock.visible = false;
    item.lock = lock;
    item.mech = { flash: 0, memory: 4, value: 0, next: 0 };
    items.push(item);
    item.group.traverse((child) => { if (child.isMesh) { child.userData.item = item; pickables.push(child); } });
    scene.add(item.group);
    const facing = BAYS[item.kind].face;
    lock.position.set(item.center.x + facing[0] * 5, item.kind === "track" ? 2.2 : 3.6, item.center.z + facing[2] * 5);
    lock.rotation.set(0, BAYS[item.kind].yaw, 0);
    lock.rotateX(-0.35);
    scene.add(lock);
    return item;
  };
  const owner = () => ({ dim: false, materials: [] });
  const ring = (radius, segments) => {
    const material = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9 });
    const made = new THREE.Mesh(new THREE.TorusGeometry(radius, 0.08, 6, segments), material);
    made.rotation.x = Math.PI / 2;
    if (segments === 4) made.rotation.z = Math.PI / 4;
    made.userData.ring = true;
    return { mesh: made, material };
  };

  const podXs = spread(menu.pilots.length, 14.4);
  menu.pilots.forEach((entry, index) => {
    const own = owner();
    const group = new THREE.Group();
    const standBuilder = builder(group);
    const steel = kit.mat("--steel3d", { owner: own });
    const dark = kit.mat("--shadow3d", { owner: own });
    const tintToken = POD_TINTS[index % POD_TINTS.length];
    const tint = kit.mat(tintToken, { owner: own });
    const design = DESIGNS[entry.id] || DESIGNS[DESIGN_ORDER[index % DESIGN_ORDER.length]];
    for (let k = 0; k < 2; k++) group.add(mesh(new THREE.CylinderGeometry(2 * P - 0.01, 2 * P - 0.01, BRICK - GAP, 32), k ? tint : steel, 0, BRICK / 2 + k * BRICK, 0));
    group.add(mesh(new THREE.CylinderGeometry(8 * P - 0.01, 8 * P - 0.01, PLATE - GAP, 56), dark, 0, 2 * BRICK + PLATE / 2, 0));
    for (let x = -8 * P; x <= 8 * P; x += P) for (let z = -8 * P; z <= 8 * P; z += P) {
      const sx = Math.round(x / P) * P + P / 2, sz = Math.round(z / P) * P + P / 2;
      const r2 = sx * sx + sz * sz;
      if (r2 < (8 * P - P * 0.6) ** 2 && r2 > (5 * P) ** 2) standBuilder.stud(dark, new THREE.Vector3(sx, STAND_TOP, sz));
    }
    standBuilder.finish();
    const rim = ring(8 * P + 0.12, 64);
    rim.mesh.position.y = 2 * BRICK + PLATE * 0.5;
    const sign = new THREE.Mesh(new THREE.PlaneGeometry(4 * P, P), tileMaterial(plate(entry.name.toUpperCase(), tintToken, 160, 40)));
    sign.position.set(0, BRICK * 1.5, 2 * P + 0.02);
    sign.userData.sign = true;
    const pod = buildPod(kit, design, own, tintToken);
    const helmetPaint = paints.find((paint) => paint.material === pod.helmet);
    let ghost = null;
    if (design.kind === "dqn" || design.kind === "bc") {
      ghost = pod.body.clone(true);
      const ghostMaterial = kit.trans("--text", { opacity: 0.2 });
      ghost.traverse((child) => { if (child.isMesh) { child.material = ghostMaterial; child.castShadow = false; } });
      ghost.rotation.order = "YZX";
      ghost.visible = false;
      ghost.userData.material = ghostMaterial;
      scene.add(ghost);
    }
    pod.extras.stack.forEach((brick, i) => { brick.visible = i < 8; });
    const yaw = -1.2;
    const hover = STAND_TOP + 1.0;
    const display = 0.8;
    pod.body.scale.setScalar(display);
    pod.body.position.set(0, hover, 0);
    pod.body.rotation.y = yaw;
    const light = new THREE.PointLight(0xffc890, 0, 18, 1.8);
    light.position.set(0, 6, 4);
    group.add(rim.mesh, sign, pod.body, light, blob(8.6 * P, 0, 0.03));
    group.position.set(podXs[index], PLATE, ROW_Z.pilot);
    register({ kind: "pilot", id: entry.id, entry, index, display, group, rim: rim.material, owner: own, pod, ghost, helmetPaint, light, yaw, hover, home: new THREE.Vector3(podXs[index], PLATE + hover, ROW_Z.pilot), center: new THREE.Vector3(podXs[index], 4, ROW_Z.pilot), lift: 0, velocity: 0, pulse: 0, grow: 1 });
  });

  const characterXs = spread(menu.podracers.length, 7.5);
  menu.podracers.forEach((entry, index) => {
    const own = owner();
    const character = buildMinifigure(kit, entry.id, own);
    const tintToken = CHARACTER_TINTS[entry.id] || POD_TINTS[index % POD_TINTS.length];
    const rim = ring(3 * P, 4);
    rim.mesh.position.y = 0.06;
    const sign = new THREE.Mesh(new THREE.PlaneGeometry(4 * P - 0.1, P * 0.95), tileMaterial(plate(entry.name.toUpperCase(), tintToken, 256, 60, true)));
    sign.position.set(0, BRICK / 2, 2 * P + 0.02);
    sign.userData.sign = true;
    character.group.add(rim.mesh, sign, blob(3 * P, 0, 0.03));
    const [cx, , cz] = bayPoint("podracer", characterXs[index]);
    character.group.position.set(cx, PLATE * 2, cz);
    character.group.rotation.y = BAYS.podracer.yaw;
    register({ kind: "podracer", id: entry.id, entry, index, group: character.group, rim: rim.material, owner: own, character, tintToken, center: new THREE.Vector3(cx, 4, cz), lift: 0, velocity: 0, pulse: 0, grow: 1 });
  });

  const gateXs = spread(menu.tracks.length, 16 * P);
  const bays = new THREE.Group();
  const bayBuild = builder(bays);
  [["podracer", "PILOTS", "--red", 26], ["track", "TRACKS", "--yellow", 40]].forEach(([kind, label, token, length]) => {
    const bay = BAYS[kind];
    const across = Math.abs(bay.along[0]) > 0.5;
    const w = across ? length : 16 * P, d = across ? 16 * P : length;
    bayBuild.brick({ x: bay.x, z: bay.z, w: w - GAP, d: d - GAP, h: PLATE - GAP, material: common("--slab3d") });
    const backX = bay.x - bay.face[0] * 7 * P, backZ = bay.z - bay.face[2] * 7 * P;
    const levels = kind === "podracer" ? 13 : 8;
    for (let k = 0; k < levels; k++) bayBuild.brick({ x: backX, y: PLATE + k * BRICK, z: backZ, w: across ? length : P * 2 - GAP, d: across ? P * 2 - GAP : length, h: BRICK - GAP, material: k === levels - 1 ? common(token) : common("--shadow3d"), top: k === levels - 1 });
    const sign = new THREE.Mesh(new THREE.PlaneGeometry(12, 1.6), tileMaterial(plate(label, token, 256, 32)));
    sign.position.set(backX + bay.face[0] * (P + 0.02), PLATE + (kind === "podracer" ? 9.2 : 4.4) * BRICK, backZ + bay.face[2] * (P + 0.02));
    sign.rotation.y = bay.yaw;
    if (kind !== "podracer") bays.add(sign);
    const mark = wordmark(kind === "podracer" ? 0.45 : 0.62);
    mark.userData.bay = kind;
    if (kind === "podracer") mark.position.set(backX, PLATE + 13 * BRICK, backZ);
    else mark.visible = false;
    mark.rotation.y = bay.yaw;
    bays.add(mark);
    if (kind === "podracer") {
      const facing = new THREE.Vector3(bay.face[0], 0, bay.face[2]);
      const along = new THREE.Vector3(bay.along[0], 0, bay.along[2]);
      const wall = new THREE.Vector3(backX, 0, backZ).addScaledVector(facing, P + 0.03);
      const tints = ["--red", "--orange", "--yellow", "--blue", "--green"];
      for (let i = 0; i < 0; i++) {
        const offset = -length / 2 + 1.2 + i * (length - 2.4) / 11;
        const pennant = new THREE.Mesh(new THREE.ConeGeometry(0.7, 1.6, 3), common(tints[i % tints.length]));
        pennant.rotation.set(Math.PI, bay.yaw, 0);
        pennant.position.copy(wall).addScaledVector(along, offset).setY(PLATE + 7.6 * BRICK);
        bays.add(pennant);
      }
      for (let i = 0; i < 5; i++) {
        const offset = -length / 2 + 2.5 + i * (length - 5) / 4;
        const lamp = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.5, 0.3, 16), kit.trans("--yellow-light", { glow: 1, opacity: 0.9 }));
        lamp.rotation.x = Math.PI / 2;
        lamp.rotation.z = bay.yaw;
        lamp.position.copy(wall).addScaledVector(along, offset).setY(PLATE + 12.3 * BRICK);
        lamp.rotation.set(0, bay.yaw, 0);
        lamp.rotateX(Math.PI / 2);
        bays.add(lamp);
      }
      const crowdMaterial = new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35 });
      paints.push({ material: crowdMaterial, token: "--silhouette3d" });
      for (let i = 0; i < 18; i++) {
        const offset = -length / 2 + 1 + i * (length - 2) / 17;
        const height = 0.9 + (i * 7919 % 5) * 0.08;
        const person = new THREE.Mesh(new THREE.PlaneGeometry(0.8, height * BRICK * 1.6), crowdMaterial);
        person.position.copy(wall).addScaledVector(along, offset).setY(PLATE + 2.6 * BRICK + height * BRICK * 0.8);
        person.rotation.y = bay.yaw;
        bays.add(person);
        const head = new THREE.Mesh(new THREE.CircleGeometry(0.36, 12), crowdMaterial);
        head.position.copy(person.position).setY(person.position.y + height * BRICK * 0.8 + 0.3);
        head.rotation.y = bay.yaw;
        bays.add(head);
      }
    }
  });
  bayBuild.finish();
  bays.userData.tag = "bays";
  scene.add(bays);
  const banners = [];
  let gateHeight = 12;
  menu.tracks.forEach((entry, index) => {
    const own = owner();
    const source = document.createElement("canvas");
    source.width = SCREEN.width;
    source.height = SCREEN.height;
    drawBanner(source, entry);
    const map = new THREE.CanvasTexture(source);
    map.magFilter = THREE.NearestFilter;
    map.minFilter = THREE.LinearFilter;
    map.generateMipmaps = false;
    banners.push({ source, map, entry });
    const gate = buildGate(kit, own, map);
    gateHeight = gate.height;
    const rimMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9 });
    const rim = new THREE.Mesh(new THREE.BoxGeometry(14 * P, 0.06, 0.25), rimMaterial);
    rim.position.set(0, 0.04, 2.6 * P);
    rim.userData.ring = true;
    gate.group.add(rim);
    for (const x of [-5 * P, 5 * P]) gate.group.add(blob(3 * P, x, 0.03));
    const [gx, , gz] = bayPoint("track", gateXs[index]);
    gate.group.position.set(gx, PLATE * 2, gz);
    gate.group.rotation.y = BAYS.track.yaw;
    register({ kind: "track", id: entry.id, entry, index, group: gate.group, rim: rimMaterial, owner: own, gate, center: new THREE.Vector3(gx, 6, gz), lift: 0, velocity: 0, pulse: 0, grow: 1 });
  });

  const spots = {};
  ["podracer", "pilot", "track"].forEach((kind) => {
    const light = new THREE.SpotLight(0xfff0d8, 0, 50, 0.55, 1, 1.4);
    light.position.set(BAYS[kind].x, 22, BAYS[kind].z + 6);
    scene.add(light, light.target);
    spots[kind] = light;
  });

  const cursor = new THREE.Group();
  const cursorMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const arrowCanvas = document.createElement("canvas");
  arrowCanvas.width = arrowCanvas.height = 64;
  const arrowPaint = arrowCanvas.getContext("2d");
  arrowPaint.fillStyle = "#ffffff";
  arrowPaint.fillRect(0, 0, 64, 64);
  arrowPaint.fillStyle = "#1a1a1a";
  arrowPaint.beginPath(); arrowPaint.moveTo(14, 20); arrowPaint.lineTo(50, 20); arrowPaint.lineTo(32, 48); arrowPaint.closePath(); arrowPaint.fill();
  cursorMaterial.map = new THREE.CanvasTexture(arrowCanvas);
  const arrowTile = new THREE.Mesh(brickGeometry(2 * P - GAP, PLATE, 2 * P - GAP, 0.03), common("--accent3d"));
  const arrowFace = new THREE.Mesh(new THREE.PlaneGeometry(2 * P - 0.12, 2 * P - 0.12), cursorMaterial);
  arrowFace.position.z = PLATE / 2 + 0.005;
  arrowTile.rotation.x = Math.PI / 2;
  arrowTile.rotation.x = 0;
  const arrowHolder = new THREE.Group();
  const arrowBody = new THREE.Mesh(brickGeometry(2 * P - GAP, 2 * P - GAP, PLATE, 0.03), common("--accent3d"));
  arrowBody.rotation.x = Math.PI / 2;
  arrowHolder.add(arrowBody, arrowFace);
  cursor.add(arrowHolder);
  cursor.visible = false;
  scene.add(cursor);

  const recorded = new THREE.Group();
  const recordedBuild = builder(recorded);
  const rollers = [];
  recordedBuild.brick({ x: DYNO.x, z: DYNO.z, w: 18 * P - GAP, d: 8 * P - GAP, h: BRICK - GAP, material: common("--shadow3d") });
  recordedBuild.brick({ x: DYNO.x, z: DYNO.z, y: -PLATE + 0.01, w: 19 * P, d: 9 * P, h: PLATE, material: common("--yellow"), top: false });
  for (const x of [-4 * P, 4 * P]) {
    const roller = mesh(new THREE.CylinderGeometry(P - 0.01, P - 0.01, 6 * P, 24), common("--steel3d", { rough: 0.2 }), DYNO.x + x, BRICK + 0.6, DYNO.z);
    roller.rotation.x = Math.PI / 2;
    recorded.add(roller);
    rollers.push(roller);
  }
  const spool = new THREE.Group();
  const drum = mesh(new THREE.CylinderGeometry(3 * P, 3 * P, 2 * P - GAP, 40), common("--yellow"), 0, 0, 0);
  drum.rotation.x = Math.PI / 2;
  spool.add(drum);
  for (let spoke = 0; spoke < 3; spoke++) {
    const bar = mesh(brickGeometry(6 * P - GAP, P - GAP, 2 * P + 0.1, 0.02), common("--yellow-light"), 0, 0, 0);
    bar.rotation.z = (spoke * Math.PI) / 3;
    bar.rotation.x = Math.PI / 2;
    spool.add(bar);
  }
  spool.position.set(REEL.x, REEL.y, REEL.z);
  recorded.add(spool);
  for (const dz of [-2 * P, 2 * P]) for (let k = 0; k < 4; k++) recordedBuild.brick({ x: REEL.x, y: k * BRICK, z: REEL.z + dz, w: P - GAP, d: P - GAP, h: BRICK - GAP, material: common("--steel3d"), top: k === 3 });
  recordedBuild.finish();
  const tapeFrom = new THREE.Vector3(REEL.x + 1.4, REEL.y + 2.3 + PLATE, REEL.z);
  const tapeTo = new THREE.Vector3(DYNO.x - 7, DYNO.y + 0.4 + PLATE, DYNO.z);
  const tape = mesh(new THREE.BoxGeometry(tapeFrom.distanceTo(tapeTo), 0.06, P * 1.6), kit.trans("--yellow-light", { glow: 0.6, opacity: 0.8 }), 0, 0, 0, false);
  tape.position.copy(tapeFrom).add(tapeTo).multiplyScalar(0.5).setY(tape.position.y - PLATE);
  tape.position.y = (tapeFrom.y + tapeTo.y) / 2 - PLATE;
  tape.rotation.z = Math.atan2(tapeTo.y - tapeFrom.y, tapeTo.x - tapeFrom.x);
  recorded.add(tape);
  recorded.position.y = PLATE;
  recorded.visible = false;
  recorded.userData.tag = "pits";
  scene.add(recorded);
  const pitsCenter = new THREE.Vector3((DYNO.x + REEL.x) / 2, 2, DYNO.z);

  const learner = new THREE.Group();
  const learnerBuild = builder(learner);
  learnerBuild.brick({ w: 6 * P - GAP, d: 6 * P - GAP, h: BRICK - GAP, material: common("--shadow3d") });
  for (let k = 1; k < 10; k++) learnerBuild.brick({ y: k * BRICK, w: 2 * P - GAP, d: 2 * P - GAP, h: BRICK - GAP, material: k % 2 ? common("--orange") : common("--steel3d"), top: false });
  learnerBuild.finish();
  const core = mesh(new THREE.ConeGeometry(2 * P, 3 * BRICK, 24), kit.trans("--orange", { glow: 0.9, opacity: 0.75 }), 0, 10 * BRICK + 1.5 * BRICK, 0);
  learner.add(core);
  const halo = new THREE.Group();
  for (let i = 0; i < 10; i++) {
    const bead = mesh(new THREE.CylinderGeometry(P / 2 - 0.02, P / 2 - 0.02, PLATE, 16), kit.trans("--orange-light", { glow: 1, opacity: 0.85 }), Math.cos((i / 10) * Math.PI * 2) * 3, 0, Math.sin((i / 10) * Math.PI * 2) * 3, false);
    halo.add(bead);
  }
  halo.position.y = 11 * BRICK;
  learner.add(halo);
  const coreLight = new THREE.PointLight(0xffa060, 1.2, 26, 1.6);
  coreLight.position.y = 11.5;
  learner.add(coreLight);
  learner.position.set(LEARNER.x, PLATE, LEARNER.z);
  learner.visible = false;
  learner.userData.tag = "learner";
  scene.add(learner);

  const packets = Array.from({ length: 24 }, () => {
    const material = new THREE.MeshPhysicalMaterial({ color: 0xffffff, roughness: 0.25, clearcoat: 0.3, envMapIntensity: 0.8, emissive: 0x000000 });
    const made = mesh(brickGeometry(P * 2 - GAP, P - GAP, BRICK - GAP, 0.03), material, 0, 0, 0, false);
    made.visible = false;
    scene.add(made);
    return { mesh: made, active: false, t: 0, duration: 1, from: new THREE.Vector3(), to: null, lift: 0 };
  });
  const launchPacket = (from, to, token, duration, lift) => {
    const packet = packets.find((candidate) => !candidate.active);
    if (!packet) return;
    Object.assign(packet, { active: true, t: 0, duration, to, lift });
    packet.from.copy(from);
    packet.mesh.material.color.copy(toy(tone(token).clone(), token));
    packet.mesh.material.emissive.copy(packet.mesh.material.color).multiplyScalar(0.25);
    packet.mesh.visible = true;
  };

  const raceBlob = blob(5);
  raceBlob.visible = false;
  scene.add(raceBlob);
  const dustCount = 140;
  const dust = new THREE.InstancedMesh(brickGeometry(P - GAP, P - GAP, PLATE - GAP, 0.02), new THREE.MeshPhysicalMaterial({ color: 0xffffff, roughness: 0.35, envMapIntensity: 0.6 }), dustCount);
  dust.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  const grains = Array.from({ length: dustCount }, () => ({ life: 0, position: new THREE.Vector3(), velocity: new THREE.Vector3(), spin: 0 }));
  const hidden = new THREE.Matrix4().makeScale(0, 0, 0);
  for (let i = 0; i < dustCount; i++) dust.setMatrixAt(i, hidden);
  dust.frustumCulled = false;
  scene.add(dust);
  let grainCursor = 0;
  const kick = (spot, speed, heat) => {
    const grain = grains[grainCursor];
    const index = grainCursor;
    grainCursor = (grainCursor + 1) % dustCount;
    grain.life = 0.35 + Math.random() * 0.25;
    grain.floor = spot.position.y + 0.05;
    grain.position.copy(spot.position).addScaledVector(spot.tangent, -5).addScaledVector(spot.right, (Math.random() - 0.5) * 5).addScaledVector(spot.normal, 0.1);
    grain.velocity.copy(spot.right).multiplyScalar((Math.random() - 0.5) * 6).addScaledVector(spot.normal, 1 + Math.random() * 2).addScaledVector(spot.tangent, speed * 0.3);
    if (grain.position.distanceTo(camera.position) < 10) { grain.life = 0; return; }
    grain.spin = Math.random() * 6;
    const spark = Math.random() < 0.08 + heat * 0.08;
    dust.setColorAt(index, toy(tone(spark ? "--orange-light" : "--ground3d").clone(), spark ? "--orange-light" : "--ground3d"));
    if (dust.instanceColor) dust.instanceColor.needsUpdate = true;
  };
  const streakCount = 48;
  const streakMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.45, depthWrite: false });
  const streaks = new THREE.InstancedMesh(new THREE.BoxGeometry(5, 0.05, 0.05), streakMaterial, streakCount);
  streaks.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  for (let i = 0; i < streakCount; i++) streaks.setMatrixAt(i, hidden);
  streaks.frustumCulled = false;
  scene.add(streaks);
  const lines = Array.from({ length: streakCount }, () => ({ life: 0, matrix: new THREE.Matrix4() }));
  let lineCursor = 0;
  const streak = (distance) => {
    const spot = place(distance + 20 + Math.random() * 40);
    const line = lines[lineCursor];
    const index = lineCursor;
    lineCursor = (lineCursor + 1) % streakCount;
    const side = Math.random() < 0.5 ? -1 : 1;
    dummy.position.copy(spot.position).addScaledVector(spot.right, side * (WIDTH / 2 - 1 + Math.random() * 5)).add(new THREE.Vector3(0, 0.6 + Math.random() * 4, 0));
    dummy.rotation.set(0, spot.heading, 0);
    dummy.scale.set(1, 1, 1);
    dummy.updateMatrix();
    line.matrix.copy(dummy.matrix);
    line.life = 0.9;
    streaks.setMatrixAt(index, line.matrix);
  };

  build.finish();

  const camera = new THREE.PerspectiveCamera(42, 2, 0.3, 3000);
  const eye = new THREE.Vector3(0, 40, GZ + 60);
  const focus = new THREE.Vector3(0, 2, GZ);
  const wantEye = new THREE.Vector3();
  const wantFocus = new THREE.Vector3();
  let width = 1, height = 1, aspect = 1;

  const state = { mode: "garage", hovered: null, focused: null, cursor: false, row: "podracer", selection: {}, race: null, view: "auto", shot: null, shots: 0, clock: 0 };

  paints.forEach(({ material: made, token }) => {
    if ((token === "--ground3d" || token === "--ridge3d") && made.isMeshPhysicalMaterial) {
      made.roughness = 0.85;
      made.clearcoat = 0;
      made.userData.env = 0.15;
    }
  });
  function recolor() {
    css = getComputedStyle(document.documentElement);
    const light = document.documentElement.dataset.resolved === "light";
    const off = tone("--off3d").clone();
    paints.forEach(({ material: made, token, owner: own, toy: toyish, glow: shine }) => {
      made.color.copy(tone(token));
      if (toyish) toy(made.color, token);
      if (own && own.dim) made.color.lerp(off, 0.72);
      if (own && own.shade) made.color.multiplyScalar(own.shade);
      if (shine && made.emissive) made.emissive.copy(made.color).multiplyScalar(shine * 0.6);
    });
    paints.forEach(({ material: made }) => { if (made.envMapIntensity !== undefined) made.envMapIntensity = made.userData.env ?? (light ? 0.35 : 0.75); });
    skyUniforms.top.value.copy(tone("--skytop3d"));
    skyUniforms.horizon.value.copy(tone("--sky3d"));
    scene.fog.color.copy(tone("--sky3d"));
    stars.visible = !light;
    second.visible = true;
    hemisphere.intensity = light ? 0.55 : 0.6;
    hemisphere.color.set(light ? 0xd8e8ff : 0x9aa8e8);
    hemisphere.groundColor.set(light ? 0xc8a878 : 0x5a3a2a);
    key.intensity = light ? 0.95 : 1.0;
    key.color.set(light ? 0xfff4e4 : 0xffb880);
    fill.intensity = light ? 0.3 : 0.45;
    scene.fog.near = light ? 320 : 260;
    scene.fog.far = light ? 1500 : 1200;
    lamps.forEach((lamp) => { lamp.intensity = 0; lamp.visible = false; });
    seats.forEach((seat, i) => bodies.setColorAt(i, toy(tone(seat.shirt).clone(), seat.shirt)));
    if (bodies.instanceColor) bodies.instanceColor.needsUpdate = true;
    circuit.kerbTones.forEach((token, i) => circuit.kerbs.setColorAt(i, toy(tone(token).clone(), token)));
    if (circuit.kerbs.instanceColor) circuit.kerbs.instanceColor.needsUpdate = true;
    cursorMaterial.color.set(0xffffff);
    streakMaterial.color.copy(tone("--text"));
    textures.forEach((draw) => draw());
    themed.forEach((callback) => callback());
    items.forEach(mark);
  }

  const glowColor = new THREE.Color();
  function mark(item) {
    const selected = state.selection[item.kind] === item.id;
    const hovered = state.hovered === item || (state.cursor && state.focused === item);
    const garage = state.mode === "garage";
    if (selected) item.rim.color.copy(tone("--accent3d"));
    else if (hovered) item.rim.color.copy(tone("--text"));
    else item.rim.color.copy(tone("--off3d"));
    item.rim.visible = garage && item.kind === state.row && (selected || hovered);
    item.lock.visible = garage && item.owner.dim && item.kind === state.row;
    glowColor.copy(tone("--accent3d")).multiplyScalar(hovered && garage && !selected ? 0.1 : 0);
    item.owner.materials.forEach((made) => made.emissive.copy(glowColor));
    if (item.gate) {
      item.gate.bulbs.forEach((bulb) => { bulb.visible = selected || hovered; });
      item.gate.strip.visible = false;
    }
    if (item.light) item.light.intensity = selected && garage ? 1.2 : 0;
  }

  function setSelection(selection, available) {
    const previous = state.selection;
    state.selection = Object.assign({}, selection);
    let changed = false;
    items.forEach((item) => {
      const dim = !available[item.kind][item.id];
      if (item.owner.dim !== dim) { item.owner.dim = dim; changed = true; }
      const shade = item.kind === state.row && state.selection[item.kind] !== item.id ? 0.72 : 1;
      if ((item.owner.shade || 1) !== shade) { item.owner.shade = shade; changed = true; }
      if (state.selection[item.kind] === item.id && previous[item.kind] !== item.id) { item.pulse = 1; item.velocity = 6; }
    });
    if (changed) recolor();
    items.forEach(mark);
    wake();
  }

  function setHover(item) {
    if (state.hovered === item) return;
    const previous = state.hovered;
    state.hovered = item;
    if (previous) mark(previous);
    if (item) mark(item);
    canvas.style.cursor = item ? "pointer" : "";
    options.onHover(item ? { kind: item.kind, id: item.id } : null);
    wake();
  }

  function setFocus(target, showCursor) {
    const item = target ? items.find((candidate) => candidate.kind === target.kind && candidate.id === target.id) : null;
    const previous = state.focused;
    state.focused = item;
    state.cursor = Boolean(showCursor && item);
    if (item) state.row = item.kind;
    if (previous) mark(previous);
    if (item) mark(item);
    wake();
  }

  let tween = null;
  function setRow(kind) {
    if (state.row !== kind && !reduced) items.forEach((item) => { if (item.kind === kind) { item.grow = 0.55 + 0.08 * item.index; } });
    if (state.row !== kind && state.mode === "garage" && primed && !reduced) {
      tween = { fromEye: eye.clone(), fromFocus: focus.clone(), t: 0, duration: 1.1, arrived: false };
      if (options.onMove) options.onMove("start");
    }
    state.row = kind;
    wake();
  }
  const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  function tweenPoint(fromEye, fromFocus, toEye, toFocus, t, outEye, outFocus) {
    const e = ease(t);
    outEye.copy(fromEye).lerp(toEye, e);
    const base = (fromEye.y + toEye.y) / 2;
    outEye.y += Math.sin(Math.PI * e) * Math.max(0.6 * base, 22 - base);
    outFocus.copy(fromFocus).lerp(toFocus, ease(Math.min(1, t / 0.4)));
  }

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const pick = (event) => {
    const bounds = canvas.getBoundingClientRect();
    pointer.set(((event.clientX - bounds.left) / bounds.width) * 2 - 1, -((event.clientY - bounds.top) / bounds.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(pickables, false).find((candidate) => candidate.object.userData.item.kind === state.row);
    return hit ? hit.object.userData.item : null;
  };
  let press = null;
  canvas.addEventListener("pointermove", (event) => {
    if (state.mode !== "garage" || event.pointerType !== "mouse") return;
    setHover(pick(event));
  });
  canvas.addEventListener("pointerleave", () => { if (state.hovered) setHover(null); });
  canvas.addEventListener("pointerdown", (event) => { press = { x: event.clientX, y: event.clientY }; });
  canvas.addEventListener("pointerup", (event) => {
    if (!press || state.mode !== "garage") return;
    const moved = Math.hypot(event.clientX - press.x, event.clientY - press.y);
    press = null;
    if (moved > 12) return;
    const item = pick(event);
    if (item) {
      state.row = item.kind;
      options.onPick(item.kind, item.id);
    }
  });

  let primed = false, wasCompact = null, snap = true;
  const band = { top: 0, bottom: 0, right: 0 };
  const shift = new THREE.Vector2();
  const light = () => document.documentElement.dataset.resolved === "light";
  const compact = () => aspect < 0.95;
  const fitter = new THREE.PerspectiveCamera();
  const corner = new THREE.Vector3();
  const frameOf = (item) => {
    if (item.frame) return item.frame;
    item.frame = new THREE.Box3();
    item.group.updateMatrixWorld(true);
    item.group.traverse((child) => {
      if (!child.isMesh || child.userData.ring) return;
      for (let node = child; node !== item.group; node = node.parent) if (!node.visible) return;
      if (!child.geometry.boundingBox) child.geometry.computeBoundingBox();
      item.frame.union(child.geometry.boundingBox.clone().applyMatrix4(child.matrixWorld));
    });
    return item.frame;
  };
  function clear(box, eyePoint, lookPoint) {
    fitter.position.copy(eyePoint);
    fitter.lookAt(lookPoint);
    fitter.updateMatrixWorld();
    fitter.matrixWorldInverse.copy(fitter.matrixWorld).invert();
    const margin = Math.min(16, height * 0.03);
    let low = false;
    for (let i = 0; i < 8; i++) {
      corner.set(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z).project(fitter);
      const x = ((corner.x + 1) / 2) * width, y = ((1 - corner.y) / 2) * height;
      if (corner.z > 1 || y < band.top + margin || x < margin || x > width - band.right - margin) return "out";
      if (y > height - band.bottom - margin) low = true;
    }
    return low ? "low" : "in";
  }
  function garageCamera() {
    const narrow = compact();
    wordmarks.forEach((mark) => { if (mark.userData.bay === "podracer") mark.visible = !narrow; });
    const bay = BAYS[state.row];
    const direction = new THREE.Vector3(bay.face[0], narrow && state.row === "podracer" ? 0.1 : bay.rise, bay.face[2]).normalize();
    camera.fov = narrow ? 50 : 36;
    const tangent = Math.tan((camera.fov * Math.PI) / 360);
    const asked = options.insets ? options.insets() : { top: 0, bottom: 0, right: 0 };
    band.top = Math.max(0, Math.min(height * 0.3, asked.top));
    band.bottom = Math.max(0, Math.min(height * 0.5, asked.bottom));
    band.right = Math.max(0, Math.min(width * 0.5, asked.right));
    const room = Math.max(0.5, (width - band.right) / Math.max(1, height - band.top - band.bottom));
    const target = state.focused && state.focused.kind === state.row ? state.focused : items.find((item) => item.kind === state.row && state.selection[item.kind] === item.id);
    const wide = { podracer: 13, pilot: 9, track: 13 }[state.row];
    const half = narrow ? Math.max({ podracer: 4.5, pilot: 7, track: 6.5 }[state.row], wide * Math.min(1, width / 1100)) : wide;
    let distance = Math.max(narrow ? 14 : 16, (half / (tangent * room)) * 1.05);
    const offset = target ? (target.center.x - bay.x) * bay.along[0] + (target.center.z - bay.z) * bay.along[2] : 0;
    const lookHeight = { podracer: 9.8, pilot: 6.2, track: 8.6 }[state.row];
    wantFocus.set(bay.x + bay.along[0] * offset, lookHeight, bay.z + bay.along[2] * offset);
    if (target) {
      fitter.fov = camera.fov;
      fitter.aspect = aspect;
      fitter.near = camera.near;
      fitter.far = camera.far;
      fitter.setViewOffset(width, height, band.right / 2, (band.bottom - band.top) / 2, width, height);
      fitter.updateProjectionMatrix();
      const box = frameOf(target);
      const lowest = state.row === "podracer" ? wantFocus.y : (box.min.y + box.max.y) / 2;
      for (let tries = 0; tries < 40; tries++) {
        wantEye.copy(wantFocus).addScaledVector(direction, distance);
        const verdict = clear(box, wantEye, wantFocus);
        if (verdict === "in") break;
        if (verdict === "low" && wantFocus.y > lowest) wantFocus.y = Math.max(lowest, wantFocus.y - 0.4);
        else distance *= 1.05;
      }
    }
    wantEye.copy(wantFocus).addScaledVector(direction, distance);
  }

  const LAP_SHOTS = ["chase", "trackside", "heli", "chase", "flyby", "trackside"];
  const PIT_SHOTS = ["pits", "reel", "screen", "pits"];
  function cut(raceState) {
    const list = raceState.style === "dyno" ? PIT_SHOTS : LAP_SHOTS;
    state.shots += 1;
    let kind = list[state.shots % list.length];
    if (kind === "flyby" && !(place(raceState.distance + 40).position.z < -80)) kind = "trackside";
    state.shot = makeShot(kind, raceState.distance, hash(`side-${state.shots}`) < 0.5 ? -1 : 1, state.clock + 5 + 3 * hash(`shot-${state.shots}`));
    snap = true;
  }
  const viewer = new THREE.PerspectiveCamera(40, 16 / 9, 0.5, 600);
  const frustum = new THREE.Frustum();
  const viewMatrix = new THREE.Matrix4();
  const markCenter = new THREE.Vector3();
  function marksInView(eye, focus, fov) {
    viewer.fov = fov;
    viewer.aspect = Math.max(1, aspect);
    viewer.updateProjectionMatrix();
    viewer.position.copy(eye);
    viewer.lookAt(focus);
    viewer.updateMatrixWorld(true);
    viewMatrix.multiplyMatrices(viewer.projectionMatrix, viewer.matrixWorldInverse);
    frustum.setFromProjectionMatrix(viewMatrix);
    let count = 0;
    wordmarks.forEach((mark) => {
      if (!mark.parent || !mark.parent.visible) return;
      mark.updateMatrixWorld(true);
      markCenter.set(0, mark.userData.height / 2, 0).applyMatrix4(mark.matrixWorld);
      if (frustum.containsPoint(markCenter) && markCenter.distanceTo(eye) < 220) count += 1;
    });
    return count;
  }
  function shotView(shot, distance) {
    const saved = state.shot;
    state.shot = shot;
    const race = state.race;
    const savedDistance = race ? race.distance : 0;
    if (race) race.distance = distance;
    raceCamera();
    if (race) race.distance = savedDistance;
    state.shot = saved;
    return { eye: wantEye.clone(), focus: wantFocus.clone(), fov: camera.fov };
  }
  function makeShot(kind, distance, side, until) {
    const build = (choice) => {
      const shot = { kind, until, side: choice };
      if (kind === "trackside" || kind === "flyby") {
        shot.at = distance + (kind === "flyby" ? 40 : 55);
        shot.spot = place(shot.at);
      }
      return shot;
    };
    const blocked = (shot) => shot.spot && blockedEye(shot.spot.position.clone().addScaledVector(shot.spot.right, shot.side * (WIDTH / 2 + 7)));
    let best = build(side);
    if (blocked(best)) best = build(-side);
    if ((kind === "trackside" || kind === "flyby" || kind === "heli") && state.race) {
      const other = build(-best.side);
      if (!blocked(other)) {
        const ahead = (shot) => { const view = shotView(shot, kind === "heli" ? distance : shot.at); return marksInView(view.eye, view.focus, view.fov); };
        if (ahead(other) > ahead(best)) best = other;
      }
    }
    return best;
  }

  const blockedEye = (p) => (Math.abs(p.x) < 46 && p.z > -14 && p.z < 12) || (Math.abs(p.x) < 33 && p.z > 33);
  const podPosition = new THREE.Vector3();
  function raceCamera() {
    band.top = band.bottom = band.right = 0;
    const race = state.race;
    const narrow = aspect < 1.3;
    if (state.view === "screen" || (state.view === "auto" && state.shot && state.shot.kind === "screen")) {
      camera.fov = narrow ? 62 : 44;
      wantFocus.copy(screenCenter).add(new THREE.Vector3(0, -4, 0));
      wantEye.set(screenCenter.x, screenCenter.y - 2, screenCenter.z + (narrow ? 64 : 46));
      return;
    }
    if (state.view === "track" || !race) {
      camera.fov = narrow ? 70 : 46;
      wantFocus.set(0, 0, -40);
      wantEye.set(0, narrow ? 220 : 150, narrow ? 190 : 130);
      return;
    }
    const pod = race.item.pod.body;
    podPosition.copy(pod.position);
    if (race.style === "dyno") {
      const kind = state.view === "chase" ? "pits" : state.shot ? state.shot.kind : "pits";
      camera.fov = narrow ? 60 : 40;
      if (kind === "reel") {
        wantEye.set(REEL.x + 4, 9, REEL.z + 22);
        wantFocus.copy(pitsCenter);
      } else {
        const angle = Math.sin(state.clock * 0.15) * 0.5;
        wantEye.set(DYNO.x + Math.sin(angle) * 30, 13, DYNO.z + Math.cos(angle) * 30);
        wantFocus.copy(pitsCenter).add(new THREE.Vector3(2, 0, 0));
      }
      return;
    }
    const spot = place(race.distance);
    if (race.mode !== "lap" || race.flight) {
      camera.fov = narrow ? 60 : 44;
      const grid = place(GRID);
      wantEye.copy(grid.position).addScaledVector(grid.right, 9).addScaledVector(grid.tangent, -20).add(new THREE.Vector3(0, 6, 0));
      wantFocus.copy(race.flight ? podPosition : grid.position).add(new THREE.Vector3(0, 1.5, 0));
      return;
    }
    const kind = state.view === "chase" ? "chase" : state.shot ? state.shot.kind : "chase";
    if (kind === "chase") {
      camera.fov = narrow ? 62 : 52;
      const behind = place(race.distance - 22), lead = place(race.distance + 24);
      wantEye.copy(behind.position).add(new THREE.Vector3(0, 8.5, 0));
      wantFocus.copy(lead.position).add(new THREE.Vector3(0, 2.5, 0));
    } else if (kind === "heli") {
      camera.fov = narrow ? 56 : 40;
      wantEye.copy(spot.position).add(new THREE.Vector3(0, 24, 0)).addScaledVector(spot.right, 16 * state.shot.side).addScaledVector(spot.tangent, -14);
      wantFocus.copy(spot.position).addScaledVector(spot.tangent, 6);
    } else {
      const anchor = state.shot.spot;
      camera.fov = narrow ? 50 : 34;
      wantEye.copy(anchor.position).addScaledVector(anchor.right, state.shot.side * (WIDTH / 2 + (kind === "flyby" ? 9 : 7))).add(new THREE.Vector3(0, kind === "flyby" ? 22 : 2.6, 0));
      wantFocus.copy(podPosition).add(new THREE.Vector3(0, 1, 0));
    }
    if (Math.abs(wantEye.x) < 46 && wantEye.z > -14 && wantEye.z < 12) wantEye.y = Math.max(wantEye.y, 18);
    if (Math.abs(wantEye.x) < 33 && wantEye.z > 33 && wantEye.z < 80) wantEye.y = Math.max(wantEye.y, 10);
    if (circuit.cliffAt && circuit.cliffAt(wantEye.x, wantEye.z) > wantEye.y - 1) wantEye.y = circuit.cliffAt(wantEye.x, wantEye.z) + 2;
  }

  function resize() {
    width = canvas.clientWidth || 1;
    height = canvas.clientHeight || 1;
    aspect = width / height;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, TIERS[tier].ratio));
    renderer.setSize(width, height, false);
    camera.aspect = aspect;
    if (state.mode === "garage") garageCamera(); else raceCamera();
    if (!primed || (state.mode === "garage" && wasCompact !== compact())) { snap = true; primed = true; }
    wasCompact = compact();
    camera.updateProjectionMatrix();
    wake();
  }

  const flame = (pod, heat, tick) => {
    pod.flames.forEach((part, index) => {
      part.visible = heat > 0;
      part.scale.x = (0.35 + heat * 1.6) * (0.85 + hash(`f-${index}-${tick}`) * 0.3);
    });
    pod.binders.forEach((binder) => {
      binder.visible = heat > 0 && hash(`b-${tick}`) > 0.12 + heat * 0.15;
    });
  };

  function cubic(points, t, out) {
    const [p0, p1, p2, p3] = points;
    const u = 1 - t;
    return out.set(0, 0, 0).addScaledVector(p0, u * u * u).addScaledVector(p1, 3 * u * u * t).addScaledVector(p2, 3 * u * t * t).addScaledVector(p3, t * t * t);
  }
  const scratch = new THREE.Vector3();
  const ahead = new THREE.Vector3();

  function launchPoints(start, end) {
    return [start, start.clone().add(new THREE.Vector3(0, 16, 0)), end.clone().add(new THREE.Vector3(0, 22, 0)), end];
  }
  function returnPoints(start, home) {
    return [start, start.clone().add(new THREE.Vector3(0, 22, 0)), home.clone().add(new THREE.Vector3(0, 16, 0)), home];
  }
  function launchEnd(style) {
    if (style === "dyno") return { end: new THREE.Vector3(DYNO.x, PLATE + DYNO.y, DYNO.z), yaw: 0 };
    const slot = place(GRID);
    return { end: slot.position.clone().addScaledVector(slot.normal, 1.4), yaw: slot.heading };
  }
  const WALK = 0.42;
  function boardPoint(from, to, t, out) {
    const mid = new THREE.Vector3(from.x + (to.x - from.x) * 0.5, from.y, from.z + (to.z - from.z) * 0.5);
    if (t < WALK) return out.copy(from).lerp(mid, t / WALK);
    const k = (t - WALK) / (1 - WALK);
    const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
    out.copy(mid).lerp(to, e);
    out.y += Math.sin(Math.PI * k) * 4;
    return out;
  }
  function boardStart(pod) {
    return new THREE.Vector3(pod.center.x - 6, PLATE + 0.02, pod.center.z + 8.5);
  }
  function seatTarget(pod) {
    return pod.pod.pilot.getWorldPosition(new THREE.Vector3());
  }
  function seat(character, pod) {
    const figure = character.character.figure;
    const rig = character.character.rig;
    if (figure.parent) figure.parent.remove(figure);
    const anchor = pod.pod.pilot;
    anchor.parent.add(figure);
    const size = 0.85 / rig.look.scale;
    figure.scale.setScalar(size);
    figure.position.set(anchor.position.x, anchor.position.y - rig.hip * size, anchor.position.z);
    figure.rotation.set(0, Math.PI / 2, 0);
    figure.visible = true;
    anchor.visible = false;
    character.seated = pod;
  }
  function flightHeading(heading, position, ahead, t, endYaw, rate = 1) {
    if (t >= 0.85) return heading + angleTo(heading, endYaw) * Math.min(1, rate * 0.8);
    const dx = ahead.x - position.x, dz = ahead.z - position.z, dy = ahead.y - position.y;
    if (dx * dx + dz * dz < 0.25 * dy * dy + 1e-6) return heading;
    return heading + angleTo(heading, Math.atan2(-dz, dx)) * rate;
  }
  function flightScale(item, label, t) {
    const grown = label === "launch" ? Math.min(1, t / 0.6) : Math.max(0, 1 - t / 0.6);
    return item.display + (1 - item.display) * (grown * grown * (3 - 2 * grown));
  }
  function racePose(item, spot, speed, bob = 0) {
    const relative = Math.max(-0.22, Math.min(0.22, spot.bank * 0.2 + spot.turn * speed * 0.04));
    const width = item.pod.width || 6;
    const reach = (item.pod.length || 16) / 2;
    const sag = reach * reach * Math.abs(spot.turn) / 2;
    const s = spot.u * LAP;
    const twist = Math.max(...[-reach, reach].map((offset) => Math.abs(Math.sin(place(s + offset).bank) - Math.sin(spot.bank))));
    const clearance = 1.4 + (width / 2) * Math.abs(Math.sin(relative)) + Math.abs(Math.sin(spot.pitch)) * 0.5 + (width / 2 + sag) * twist;
    return {
      position: spot.position.clone().addScaledVector(spot.normal, clearance + bob),
      roll: spot.bank + relative,
      heading: spot.heading,
      pitch: spot.pitch,
    };
  }
  function fly(raceState, points, duration, endYaw, done) {
    raceState.flight = { points, duration: reduced ? 0.01 : duration, t: 0, endYaw, done };
  }

  function race(selection) {
    if (!pinned && tier > 0) Object.assign(probe, { samples: [], after: 0, done: false });
    const item = items.find((candidate) => candidate.kind === "pilot" && candidate.id === selection.pilot);
    if (!item) return;
    state.mode = "race";
    state.view = aspect < 1 ? "chase" : "auto";
    state.shot = null;
    cursor.visible = false;
    setHover(null);
    items.forEach(mark);
    const podracer = selection.podracer;
    const style = podracer === "quadinaros" ? "dyno" : podracer === "sebulba" ? "split" : "fused";
    if (item.helmetPaint && item.helmetPaint.token !== CHARACTER_TINTS[podracer]) {
      item.helmetPaint.token = CHARACTER_TINTS[podracer] || "--text";
      recolor();
    }
    const body = item.pod.body;
    body.rotation.order = "YZX";
    const start = body.getWorldPosition(new THREE.Vector3());
    const raceState = { item, style, mode: "launch", distance: GRID, speed: 0, target: 0, heat: 0, training: false, flight: null, emit: 0, laps: 0, goAt: null, dust: 0, lines: 0, flash: 0, memory: 4, value: 0, syncAt: 0, ghostFlash: 0 };
    if (item.ghost) item.ghost.visible = false;
    item.pod.extras.stack.forEach((brick, i) => { brick.visible = i < 4; });
    state.race = raceState;
    recorded.visible = style === "dyno";
    learner.visible = style === "split";
    door.visible = false;
    items.forEach((other) => { if (other.kind === "track") other.group.visible = false; });
    const yaw = body.rotation.y;
    item.group.remove(body);
    scene.add(body);
    body.position.copy(start);
    body.rotation.set(0, yaw, 0);
    const { end, yaw: endYaw } = launchEnd(style);
    const points = launchPoints(start, end);
    fly(raceState, points, 2.8, endYaw, () => { raceState.mode = style === "dyno" ? "dyno" : "grid"; });
    circuit.setLights(0, false);
    snap = true;
    raceCamera();
    wake();
  }

  function back() {
    const raceState = state.race;
    if (!raceState) return Promise.resolve();
    return new Promise((resolve) => {
      raceState.target = 0;
      raceState.training = false;
      const body = raceState.item.pod.body;
      const start = body.position.clone();
      const home = raceState.item.home.clone();
      const points = returnPoints(start, home);
      raceState.mode = "return";
      state.mode = "returning";
      state.view = "track";
      fly(raceState, points, 2.4, raceState.item.yaw, () => {
        scene.remove(body);
        raceState.item.group.add(body);
        body.position.set(0, raceState.item.hover, 0);
        body.scale.setScalar(raceState.item.display);
        body.rotation.set(0, raceState.item.yaw, 0);
        if (raceState.item.ghost) raceState.item.ghost.visible = false;
        raceState.item.pod.extras.stack.forEach((brick, i) => { brick.visible = i < 8; });
        recorded.visible = false;
        learner.visible = false;
        door.visible = true;
        items.forEach((other) => { if (other.kind === "track") other.group.visible = true; });
        packets.forEach((packet) => { packet.active = false; packet.mesh.visible = false; });
        grains.forEach((grain, i) => { grain.life = 0; dust.setMatrixAt(i, hidden); });
        lines.forEach((line, i) => { line.life = 0; streaks.setMatrixAt(i, hidden); });
        dust.instanceMatrix.needsUpdate = streaks.instanceMatrix.needsUpdate = true;
        circuit.setLights(0, false);
        state.race = null;
        raceBlob.visible = false;
        state.mode = "garage";
        state.view = "auto";
        snap = true;
        items.forEach(mark);
        resolve();
      });
      wake();
    });
  }

  function setSpeed(sps) {
    if (!state.race) return;
    state.race.target = speedFor(sps);
    state.race.heat = Math.max(0, Math.min(1, (Math.log10(Math.max(1, sps)) - 2.5) / 2.5));
  }
  function setTraining(training) {
    if (!state.race) return;
    state.race.training = training;
    if (!training) state.race.target = 0;
  }
  function pulse(update) {
    const raceState = state.race;
    if (!raceState || document.hidden) return;
    const extras = raceState.item.pod.extras;
    raceState.flash = 1;
    extras.bars.forEach((bar) => { bar.scale.y = 0.2 + Math.random() * 1.2; });
    if (extras.stack.length) raceState.memory = Math.min(extras.stack.length, raceState.memory + 1);
    if (extras.needle && update) raceState.value = Math.min(1, update.update / 120);
  }
  function setView(view) {
    state.view = view;
    state.shot = null;
    snap = true;
    if (state.mode === "race") raceCamera();
    wake();
  }

  function mechanics(extras, live, dt, clock, speed) {
    live.flash = Math.max(0, live.flash - dt * 3);
    extras.clamps.forEach((clamp) => clamp.material.emissive.copy(tone("--yellow-light")).multiplyScalar(live.flash * 0.9));
    extras.gauges.forEach((gauge, i) => gauge.material.emissive.setScalar(live.flash * (0.15 + 0.1 * i)));
    if (extras.needle) extras.needle.rotation.y += ((Math.PI * 0.75 - live.value * Math.PI * 1.5) - extras.needle.rotation.y) * Math.min(1, dt * 4);
    extras.stack.forEach((brick, i) => { brick.visible = i < live.memory; });
    if (extras.crane) { extras.crane.rotation.y = Math.sin(clock * 1.7) * 0.9; extras.crane.position.y = 4 * BRICK + Math.abs(Math.sin(clock * 3.4)) * 0.2; }
    if (extras.reel) extras.reel.rotation.z -= dt * (0.5 + speed * 0.15);
  }
  function showcase(item, dt, clock) {
    const live = item.mech;
    if (clock > live.next) {
      live.next = clock + 0.9;
      live.flash = 1;
      item.pod.extras.bars.forEach((bar) => { bar.scale.y = 0.2 + Math.random() * 1.2; });
      live.memory = live.memory >= item.pod.extras.stack.length ? 4 : live.memory + 1;
      live.value = 0.5 + 0.45 * Math.sin(clock * 0.9);
    }
    mechanics(item.pod.extras, live, dt, clock, 4);
  }
  function cheer(id) {
    const item = items.find((candidate) => candidate.kind === "podracer" && candidate.id === id);
    if (item) item.cheer = state.clock + 1.1;
    wake();
  }
  let boarding = null;
  function board(podracer, pilot) {
    const character = items.find((candidate) => candidate.kind === "podracer" && candidate.id === podracer);
    const pod = items.find((candidate) => candidate.kind === "pilot" && candidate.id === pilot);
    if (!character || !pod) return Promise.resolve();
    return new Promise((resolve) => {
      const from = boardStart(pod);
      boarding = { character, pod, from, t: 0, duration: reduced ? 0.01 : 1.15, done: resolve };
      wake();
    });
  }
  function unboard() {
    items.forEach((item) => {
      if (item.kind === "podracer") {
        const figure = item.character.figure;
        if (figure.parent !== item.group) { if (figure.parent) figure.parent.remove(figure); item.group.add(figure); }
        figure.position.set(0, item.character.base, 0);
        figure.rotation.set(0, 0, 0);
        figure.visible = true;
        figure.scale.setScalar(1);
        if (item.seated) item.seated.pod.pilot.visible = true;
        item.seated = null;
      }
      if (item.kind === "pilot" && item.helmetPaint) item.helmetPaint.token = "--text";
    });
    boarding = null;
    recolor();
    wake();
  }
  function step(dt, tick, motion = dt) {
    state.clock += dt;
    const clock = state.clock;
    const garage = state.mode === "garage";
    items.forEach((item) => {
      const selected = state.selection[item.kind] === item.id;
      const hovered = state.hovered === item || (state.cursor && state.focused === item);
      const target = selected && garage ? 1 : 0;
      for (let left = dt; left > 0; left -= 1 / 120) {
        const h = Math.min(left, 1 / 120);
        item.velocity += ((target - item.lift) * 90 - item.velocity * 9) * h;
        item.lift += item.velocity * h;
      }
      item.pulse = Math.max(0, item.pulse - dt * 3);
      const click = 1 + Math.sin(item.pulse * Math.PI) * 0.06;
      item.grow += ((hovered && garage ? 1.04 : 1) - item.grow) * Math.min(1, dt * (item.grow < 0.95 ? 5 : 10));
      if (item.kind === "pilot" && item.pod.body.parent === item.group) {
        const body = item.pod.body;
        body.position.y = item.hover + item.lift * 1.2 + 0.12 * Math.sin(clock * 2 + item.index);
        if (selected && !reduced) body.rotation.y += dt * 0.35;
        else body.rotation.y += angleTo(body.rotation.y, item.yaw) * Math.min(1, dt * 2);
        item.feature = (item.feature ?? 1) + (((selected || state.row !== "pilot") ? 1 : 0.78) - (item.feature ?? 1)) * Math.min(1, dt * 5);
        body.scale.setScalar(item.display * item.grow * click * item.feature);
        flame(item.pod, selected && garage ? 0.25 : 0, tick);
        if (selected && garage && state.row === "pilot" && !reduced) showcase(item, dt, clock);
      }
      if (item.kind === "podracer") {
        const { figure, arms, base, wobble } = item.character;
        const cheering = item.cheer && clock < item.cheer && !reduced;
        if (figure.parent === item.group) {
          const hop = cheering ? Math.abs(Math.sin(clock * 9)) * 0.6 : selected && !reduced ? Math.abs(Math.sin(clock * 3)) * 0.3 * Math.min(1, Math.abs(item.lift)) : 0;
          figure.position.y = base + Math.max(0, item.lift) * STUD.height * 3 + hop;
          figure.position.z = Math.max(0, item.lift) * 1.2;
        }
        const walking = Boolean(boarding && boarding.character === item && boarding.t < WALK);
        const jumping = Boolean(boarding && boarding.character === item && boarding.t >= WALK);
        const seated = Boolean(item.seated) || jumping;
        pose(item.character.rig, { clock, selected: selected && garage && !seated, hovered, cheering, walking, seated, reduced, lift: item.lift });
        if (!walking && !seated && figure.parent === item.group) figure.rotation.y = hovered ? Math.sin(clock * 2) * 0.15 : 0;
        figure.rotation.z = wobble && !reduced ? Math.sin(clock * 5.3) * 0.05 + Math.sin(clock * 8.1) * 0.02 : 0;
        item.group.scale.setScalar(item.grow * click * (1 + 0.06 * Math.max(0, item.lift)));
      }
      if (item.kind === "track") {
        item.gate.group.position.y = PLATE + Math.max(0, item.lift) * STUD.height * 2;
        if (!item.basePosition) item.basePosition = item.gate.group.position.clone();
        const face = BAYS.track.face;
        item.gate.group.position.x = item.basePosition.x + face[0] * Math.max(0, item.lift) * BRICK;
        item.gate.group.position.z = item.basePosition.z + face[2] * Math.max(0, item.lift) * BRICK;
        item.gate.banner.material.emissiveIntensity = selected ? 1 : 0.35;
        item.group.scale.setScalar(item.grow * click);
        if (selected) item.gate.bulbs.forEach((bulb, index) => { bulb.visible = reduced || (Math.floor(clock * 6) + index) % 3 !== 0; });
      }
      if (selected && garage && item.kind !== state.row) spots[item.kind].intensity = 0;
      if (selected && garage && item.kind === state.row) {
        const spot = spots[item.kind];
        spot.intensity = light() ? 0.6 : 1.2;
        spot.angle = 0.38;
        spot.penumbra = 0.8;
        const face = BAYS[item.kind].face;
        spot.position.set(item.center.x + face[0] * 8, 24, item.center.z + face[2] * 8);
        spot.target.position.set(item.center.x, 0, item.center.z);
      }
    });
    items.forEach((item) => {
      if (item.kind === "podracer" || item.kind === "track") item.group.visible = state.mode === "garage" ? item.kind === state.row : item.kind === "podracer" ? false : item.group.visible;
      if (item.kind === "track" && state.mode !== "garage") item.group.visible = false;
    });
    bays.visible = state.mode === "garage";
    if (boarding) {
      const { character, pod, from } = boarding;
      const figure = character.character.figure;
      if (figure.parent !== scene) {
        if (figure.parent) figure.parent.remove(figure);
        scene.add(figure);
        figure.position.copy(from);
      }
      boarding.t = Math.min(1, boarding.t + motion / boarding.duration);
      const to = seatTarget(pod);
      const t = boarding.t;
      boardPoint(from, to, t, figure.position);
      figure.rotation.set(0, Math.atan2(to.x - from.x, to.z - from.z), 0);
      const shrink = 0.85 / character.character.rig.look.scale * pod.pod.body.scale.x;
      figure.scale.setScalar(t < WALK ? 1 : 1 + (shrink - 1) * ((t - WALK) / (1 - WALK)));
      if (t >= 1) {
        seat(character, pod);
        if (pod.helmetPaint) { pod.helmetPaint.token = CHARACTER_TINTS[character.id] || "--text"; recolor(); }
        pod.pulse = 1;
        pod.velocity = 6;
        const done = boarding.done;
        boarding = null;
        done();
      }
    }
    if (!garage) Object.values(spots).forEach((spot) => { spot.intensity = 0; });
    const podFocus = state.row === "pilot" ? (state.focused && state.focused.kind === "pilot" ? state.focused : items.find((item) => item.kind === "pilot" && state.selection.pilot === item.id)) : null;
    if (podFocus) baySign.position.x += (Math.max(-24, Math.min(24, podFocus.center.x)) - baySign.position.x) * Math.min(1, dt * 3);
    if (state.cursor && state.focused && garage) {
      const item = state.focused;
      cursor.visible = true;
      const bounce = Math.abs(Math.sin(clock * 4)) * 0.6;
      if (item.kind === "pilot") {
        cursor.position.set(item.center.x, 1.4 + bounce, item.center.z + 8.6 * P);
        cursor.lookAt(camera.position.x, cursor.position.y, camera.position.z);
        cursor.rotateZ(Math.PI);
      } else {
        const top = item.kind === "track" ? gateHeight + 1.6 : 10;
        cursor.position.set(item.center.x, top + bounce, item.center.z);
        cursor.lookAt(camera.position.x, cursor.position.y, camera.position.z);
      }
      cursorMaterial.color.set(0xffffff);
    } else cursor.visible = false;

    circuit.flags.forEach((flag) => { flag.holder.rotation.y = flag.base + (reduced ? 0 : Math.sin(clock * 3 + flag.phase) * 0.3); });

    const raceState = state.race;
    if (raceState) {
      const body = raceState.item.pod.body;
      const ease = 1 - Math.exp(-motion * (raceState.target > raceState.speed ? 0.9 : 1.6));
      raceState.speed += (raceState.target - raceState.speed) * ease;
      const throttle = Math.min(1, raceState.speed / Math.max(1, raceState.target || 1));
      if (raceState.style !== "dyno" && state.mode === "race") {
        if (!raceState.training) circuit.setLights(1 + (Math.floor(clock * 1.6) % 3), false);
        else {
          if (raceState.goAt === null) raceState.goAt = clock;
          circuit.setLights(clock - raceState.goAt < 2.2 ? 4 : 0, clock - raceState.goAt < 2.2);
        }
      }
      if (raceState.flight) {
        const flight = raceState.flight;
        flight.t = Math.min(1, flight.t + motion / flight.duration);
        const t = flight.t * flight.t * (3 - 2 * flight.t);
        cubic(flight.points, t, body.position);
        cubic(flight.points, Math.min(1, t + 0.02), ahead);
        body.rotation.y = flightHeading(body.rotation.y, body.position, ahead, flight.t, flight.endYaw, Math.min(1, dt * 8));
        body.scale.setScalar(flightScale(raceState.item, raceState.mode === "return" ? "return" : "launch", flight.t));
        body.rotation.x *= 0.9;
        body.rotation.z *= 0.9;
        flame(raceState.item.pod, 0.7, tick);
        if (flight.t >= 1) {
          raceState.flight = null;
          body.rotation.y = flight.endYaw;
          flight.done();
        }
      } else if (raceState.mode === "grid" || raceState.mode === "lap") {
        const moving = raceState.training && raceState.speed > 0.05;
        if (moving) raceState.mode = "lap";
        if (raceState.mode === "lap") raceState.distance += raceState.speed * motion;
        const spot = place(raceState.distance);
        const heat = raceState.heat * throttle;
        const wobble = reduced ? 0 : Math.sin(clock * 23) * 0.03 * heat;
        const pose = racePose(raceState.item, spot, raceState.speed, 0.12 * Math.sin(clock * 9) + Math.abs(wobble) * 3);
        body.position.copy(pose.position);
        body.rotation.set(pose.roll + wobble, pose.heading, pose.pitch, "YZX");
        raceState.laps = Math.max(0, Math.floor((raceState.distance - GRID) / LAP));
        flame(raceState.item.pod, raceState.mode === "lap" ? Math.max(0.15, heat) : 0.1, tick);
        if (raceState.mode === "lap" && !reduced) {
          raceState.dust += dt * raceState.speed * 0.9;
          while (raceState.dust > 1) { raceState.dust -= 1; kick(spot, raceState.speed, heat); }
          if (heat > 0.55) {
            raceState.lines += dt * 30 * heat;
            while (raceState.lines > 1) { raceState.lines -= 1; streak(raceState.distance); }
          }
        }
      } else if (raceState.mode === "dyno") {
        body.position.set(DYNO.x, PLATE + DYNO.y + 0.04 * Math.sin(clock * 20 * raceState.heat), DYNO.z);
        body.rotation.set(0, 0, 0);
        const turn = raceState.speed * dt * 1.2;
        rollers.forEach((roller) => { roller.rotation.y -= turn; });
        spool.rotation.z += turn * 0.25;
        flame(raceState.item.pod, raceState.training ? Math.max(0.15, raceState.heat) * throttle : 0.08, tick);
        if (raceState.training && raceState.speed > 1) {
          raceState.emit -= dt;
          if (raceState.emit <= 0) {
            raceState.emit = 0.9 / (1 + raceState.speed * 0.12);
            launchPacket(tapeFrom, () => tapeTo, "--yellow-light", 0.9, 0);
          }
        }
      } else if (raceState.mode === "return") {
        flame(raceState.item.pod, 0.3, tick);
        body.rotation.x *= 0.9;
        body.rotation.z *= 0.9;
      }
      const extras = raceState.item.pod.extras;
      mechanics(extras, raceState, dt, clock, raceState.speed);
      const ghost = raceState.item.ghost;
      if (ghost) {
        const racing = raceState.mode === "lap" && !raceState.flight;
        ghost.visible = racing;
        if (racing) {
          let offset;
          if (extras.kind === "dqn") {
            if (clock - raceState.syncAt > 4) { raceState.syncAt = clock; raceState.ghostFlash = 1; }
            offset = -(1.5 + (clock - raceState.syncAt) * 5);
          } else offset = 18 + Math.sin(clock * 0.7) * 3;
          raceState.ghostFlash = Math.max(0, raceState.ghostFlash - dt * 2.5);
          const ghostSpot = place(raceState.distance + offset);
          ghost.position.copy(ghostSpot.position).addScaledVector(ghostSpot.normal, 0.9 + 0.12 * Math.sin(clock * 9 + 1));
          ghost.rotation.set(ghostSpot.bank * 1.2, ghostSpot.heading, ghostSpot.pitch, "YZX");
          ghost.scale.copy(body.scale);
          ghost.userData.material.opacity = 0.16 + raceState.ghostFlash * 0.5;
        }
      }
      raceBlob.visible = !raceState.flight || raceState.flight.t > 0.8;
      if (raceBlob.visible) {
        const below = raceState.mode === "dyno" ? { position: new THREE.Vector3(DYNO.x, PLATE + BRICK + 0.05, DYNO.z) } : place(raceState.distance);
        raceBlob.position.copy(below.position).add(new THREE.Vector3(0, 0.06, 0));
        raceBlob.scale.setScalar(raceState.mode === "dyno" ? 1 : 1.2);
      }
      if (raceState.style === "split") halo.rotation.y += dt * (0.4 + (raceState.training ? raceState.speed * 0.05 : 0));
      if (state.view === "auto" && state.mode === "race") {
        const shot = state.shot;
        const ready = raceState.style === "dyno" || (raceState.mode === "lap" && !raceState.flight);
        if (ready && (!shot || clock > shot.until || (shot.at !== undefined && raceState.distance > shot.at + 22))) cut(raceState);
      }
    }

    let moved = false;
    grains.forEach((grain, i) => {
      if (grain.life <= 0) return;
      grain.life -= dt;
      grain.velocity.y -= 14 * dt;
      grain.position.addScaledVector(grain.velocity, dt);
      if (grain.position.y < grain.floor) { grain.position.y = grain.floor; grain.velocity.multiplyScalar(0.4); }
      if (grain.life <= 0) dust.setMatrixAt(i, hidden);
      else {
        dummy.position.copy(grain.position);
        dummy.rotation.set(grain.spin * grain.life, grain.spin, 0);
        dummy.scale.setScalar(0.45 * Math.min(1, grain.life * 4));
        dummy.updateMatrix();
        dust.setMatrixAt(i, dummy.matrix);
      }
      moved = true;
    });
    dummy.scale.set(1, 1, 1);
    dummy.rotation.set(0, 0, 0);
    if (moved) dust.instanceMatrix.needsUpdate = true;
    let streaked = false;
    lines.forEach((line, i) => {
      if (line.life <= 0) return;
      line.life -= dt;
      if (line.life <= 0) streaks.setMatrixAt(i, hidden);
      streaked = true;
    });
    if (streaked) streaks.instanceMatrix.needsUpdate = true;

    packets.forEach((packet) => {
      if (!packet.active) return;
      packet.t += dt / packet.duration;
      if (packet.t >= 1) { packet.active = false; packet.mesh.visible = false; return; }
      packet.mesh.position.copy(packet.from).lerp(packet.to(), packet.t);
      packet.mesh.position.y += Math.sin(packet.t * Math.PI) * packet.lift;
      packet.mesh.rotation.y += dt * 4;
    });

    if (!reduced && tick % 2 === 0) placeCrowd(clock, state.race && state.race.mode === "lap");

    if (state.mode === "race") raceCamera();
    else if (state.mode === "returning") raceCamera();
    else garageCamera();
    const fit = state.mode === "garage" ? BAYS[state.row] : null;
    const extent = fit ? 34 : 90;
    const center = fit ? new THREE.Vector3(fit.x, 0, fit.z) : new THREE.Vector3(0, 0, -30);
    if (key.shadow.camera.right !== extent || !key.target.position.equals(center)) {
      key.target.position.copy(center);
      key.position.copy(center).add(new THREE.Vector3(-70, 120, 70));
      Object.assign(key.shadow.camera, { left: -extent, right: extent, top: extent, bottom: -extent });
      key.shadow.camera.updateProjectionMatrix();
    }
    const chasing = state.mode === "race" && state.race && state.race.mode === "lap" && (state.view === "chase" || (state.view === "auto" && state.shot && state.shot.kind === "chase"));
    const rate = chasing ? 8 : state.shot && (state.shot.kind === "trackside" || state.shot.kind === "flyby") ? 12 : 2.4;
    const follow = reduced || snap ? 1 : 1 - Math.exp(-Math.min(0.6, motion) * rate);
    if (tween && state.mode === "garage" && !snap) {
      tween.t = Math.min(1, tween.t + motion / tween.duration);
      tweenPoint(tween.fromEye, tween.fromFocus, wantEye, wantFocus, tween.t, eye, focus);
      if (!tween.arrived && tween.t >= 0.75) { tween.arrived = true; if (options.onMove) options.onMove("arrive"); }
      if (tween.t >= 1) tween = null;
    } else {
      if (tween) { tween = null; if (options.onMove) options.onMove("arrive"); }
      eye.lerp(wantEye, follow);
      focus.lerp(wantFocus, follow);
    }
    scene.fog.near = state.mode === "garage" ? 150 : light() ? 320 : 260;
    scene.fog.far = state.mode === "garage" ? 800 : light() ? 1500 : 1200;
    shift.x += (band.right / 2 - shift.x) * follow;
    shift.y += ((band.bottom - band.top) / 2 - shift.y) * follow;
    if (Math.abs(shift.x) > 0.5 || Math.abs(shift.y) > 0.5) camera.setViewOffset(width, height, shift.x, shift.y, width, height);
    else camera.clearViewOffset();
    snap = false;
    camera.position.copy(eye);
    camera.lookAt(focus);
    camera.updateProjectionMatrix();
  }

  const projected = new THREE.Vector3();
  function project(point) {
    projected.copy(point).project(camera);
    return { x: ((projected.x + 1) / 2) * width, y: ((1 - projected.y) / 2) * height, visible: projected.z < 1 && projected.z > -1 };
  }
  function anchors() {
    const result = {};
    if (state.race) {
      const pod = state.race.item.pod.body;
      result.pod = project(scratch.copy(pod.position).add(new THREE.Vector3(0, 4, 0)));
      if (state.race.style === "split") result.learner = project(new THREE.Vector3(LEARNER.x, 17, LEARNER.z));
      if (state.race.style === "dyno") result.reel = project(new THREE.Vector3(REEL.x, REEL.y + 4.4, REEL.z));
    }
    return result;
  }

  let frameHandle = 0, last = 0, lastRender = 0, tick = 0, onScreen = true, lost = false, screenDirty = false, stirred = 0, rested = 0;
  const probe = { samples: [], after: 0, done: pinned };
  const active = () => !lost && onScreen && !document.hidden;
  const casters = () => scene.traverse((object) => { if (object.userData.studs) object.castShadow = TIERS[tier].studs; });
  function degrade() {
    tier -= 1;
    casters();
    const { shadow, soft } = TIERS[tier];
    key.shadow.mapSize.set(shadow, shadow);
    if (key.shadow.map) { key.shadow.map.dispose(); key.shadow.map = null; }
    if (renderer.shadowMap.type !== (soft ? THREE.PCFSoftShadowMap : THREE.PCFShadowMap)) {
      renderer.shadowMap.type = soft ? THREE.PCFSoftShadowMap : THREE.PCFShadowMap;
      scene.traverse((object) => { if (object.material) [].concat(object.material).forEach((made) => { made.needsUpdate = true; }); });
    }
    resize();
  }
  function measure(interval, now) {
    if (probe.done) return;
    if (!probe.after) probe.after = now + 1200;
    if (now < probe.after) return;
    probe.samples.push(interval);
    const slow = probe.samples.length >= 4 && Math.min(...probe.samples) > 120;
    if (!slow && probe.samples.length < 12) return;
    const median = probe.samples.sort((x, y) => x - y)[6];
    probe.samples = [];
    if ((slow || median > 45) && tier > 0) { degrade(); probe.after = now + 600; } else probe.done = true;
  }
  const still = () => reduced && state.mode === "garage" && !boarding && !tween && eye.distanceToSquared(wantEye) < 1e-6 && focus.distanceToSquared(wantFocus) < 1e-6
    && items.every((item) => Math.abs(item.velocity) < 1e-3 && item.pulse < 1e-3 && Math.abs(item.grow - (state.hovered === item || (state.cursor && state.focused === item) ? 1.04 : 1)) < 1e-3);
  function loop(now) {
    frameHandle = 0;
    if (!active()) return;
    const busy = state.mode !== "garage" || state.hovered || state.cursor || boarding || tween;
    const interval = busy ? 1000 / 60 : coarse || now - stirred > 8000 ? 1000 / 30 : 1000 / 40;
    if (now - lastRender >= interval - 2) {
      const raw = (now - (last || now)) / 1000;
      const dt = Math.min(0.1, raw);
      if (lastRender) measure(now - lastRender, now);
      last = now;
      lastRender = now;
      tick += 1;
      step(dt, tick, Math.min(1, raw));
      if (screenDirty) { screenTexture.needsUpdate = true; screenDirty = false; }
      renderer.render(scene, camera);
      options.onRender(anchors());
      rested = still() ? rested + 1 : 0;
      if (rested > 2) return;
    }
    frameHandle = requestAnimationFrame(loop);
  }
  function wake() {
    stirred = performance.now();
    rested = 0;
    if (!frameHandle && active()) {
      last = performance.now();
      lastRender = 0;
      frameHandle = requestAnimationFrame(loop);
    }
  }
  document.addEventListener("visibilitychange", wake);
  new IntersectionObserver(([entry]) => { onScreen = entry.isIntersecting; wake(); }).observe(canvas);
  new ResizeObserver(resize).observe(canvas);
  canvas.addEventListener("webglcontextlost", (event) => {
    event.preventDefault();
    lost = true;
    options.onLost();
  });

  function refreshText() {
    banners.forEach((banner) => { drawBanner(banner.source, banner.entry); banner.map.needsUpdate = true; });
    textures.forEach((draw) => draw());
    wake();
  }

  function probes() {
    return {
      THREE,
      scene,
      camera,
      renderer,
      quality: () => ({ tier, ratio: renderer.getPixelRatio(), shadow: key.shadow.mapSize.x, calls: renderer.info.render.calls, triangles: renderer.info.render.triangles, geometries: renderer.info.memory.geometries, textures: renderer.info.memory.textures, rendering: Boolean(frameHandle) }),
      snapshot(view) {
        const saved = renderer.getPixelRatio();
        renderer.setPixelRatio(1);
        renderer.setSize(view.width, view.height, false);
        renderer.render(scene, view.camera);
        const data = renderer.domElement.toDataURL("image/png");
        renderer.setPixelRatio(saved);
        resize();
        return data;
      },
      items,
      circuit,
      constants: { GRID, DYNO, BAYS, STAND_TOP, PLATE, WIDTH },
      cubic,
      launchPoints,
      returnPoints,
      flightHeading,
      flightScale,
      racePose,
      launchEnd,
      boardPoint,
      boardStart,
      seatTarget,
      tweenPoint,
      wordmarks,
      marksInView,
      garageView(row, width, height) {
        const saved = { row: state.row, mode: state.mode, focused: state.focused, cursor: state.cursor };
        state.mode = "garage";
        state.row = row;
        const results = items.filter((item) => item.kind === row).map((item) => {
          state.focused = item;
          state.cursor = true;
          garageCamera();
          const view = new THREE.PerspectiveCamera(camera.fov, width / height, 0.5, 3000);
          view.position.copy(wantEye);
          view.lookAt(wantFocus);
          if (band.right > 0.5 || Math.abs(band.bottom - band.top) > 0.5) view.setViewOffset(width, height, band.right / 2, (band.bottom - band.top) / 2, width, height);
          view.updateMatrixWorld(true);
          view.updateProjectionMatrix();
          return { id: item.id, item, camera: view };
        });
        Object.assign(state, saved);
        return results;
      },
      cursorPose(item, camera) {
        const holder = new THREE.Object3D();
        if (item.kind === "pilot") holder.position.set(item.center.x, 1.4 + 0.6, item.center.z + 8.6 * P);
        else holder.position.set(item.center.x, (item.kind === "track" ? gateHeight + 1.6 : 10) + 0.6, item.center.z);
        holder.lookAt(camera.position.x, holder.position.y, camera.position.z);
        holder.updateMatrixWorld(true);
        return holder.matrixWorld.clone();
      },
      cursorObject: cursor,
      podFocusSign(item) { baySign.position.x = Math.max(-24, Math.min(24, item.center.x)); baySign.updateMatrixWorld(true); },
      garageEye(row) {
        const saved = { row: state.row, mode: state.mode, focused: state.focused, cursor: state.cursor };
        state.mode = "garage";
        state.row = row;
        const results = items.filter((item) => item.kind === row).map((item) => {
          state.focused = item;
          state.cursor = true;
          garageCamera();
          return { id: item.id, eye: wantEye.clone(), focus: wantFocus.clone() };
        });
        Object.assign(state, saved);
        return results;
      },
      raceEye(kind, distance, side, style, item) {
        const saved = { race: state.race, shot: state.shot, view: state.view, mode: state.mode };
        const body = item.pod.body;
        const savedPosition = body.position.clone();
        const spot = place(distance);
        body.position.copy(spot.position).addScaledVector(spot.normal, 0.9);
        state.race = { item, style, mode: style === "dyno" ? "dyno" : "lap", distance, flight: null };
        state.mode = "race";
        state.view = kind === "screen" || kind === "track" ? kind : kind === "chase-view" ? "chase" : "auto";
        state.shot = makeShot(kind, distance, side, Infinity);
        raceCamera();
        const result = { eye: wantEye.clone(), focus: wantFocus.clone(), pod: body.position.clone(), fov: camera.fov, marks: marksInView(wantEye, wantFocus, camera.fov) };
        body.position.copy(savedPosition);
        Object.assign(state, saved);
        return result;
      },
    };
  }

  casters();
  recolor();
  resize();

  return {
    probes,
    setSelection,
    setFocus,
    setRow,
    cheer,
    board,
    unboard,
    race,
    back,
    setSpeed,
    setTraining,
    setView,
    pulse,
    get view() { return state.view; },
    get laps() { return state.race ? state.race.laps : 0; },
    get compact() { return compact(); },
    screenChanged() { screenDirty = true; },
    theme() { recolor(); wake(); },
    refreshText,
    resize,
    wake,
  };
}
