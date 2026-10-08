import { P, BRICK, PLATE, builder, brickGeometry, bake } from "./bricks.js";

const T = () => window.THREE;
const MM = 0.1;

function mesh(geometry, material, x = 0, y = 0, z = 0) {
  const made = new (T().Mesh)(geometry, material);
  made.position.set(x, y, z);
  made.castShadow = true;
  made.receiveShadow = true;
  return made;
}

function flat(made) {
  made.rotation.x = Math.PI / 2;
  return made;
}

function canvasTexture(width, height, paint) {
  const THREE = T();
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const map = new THREE.CanvasTexture(canvas);
  map.anisotropy = 4;
  const draw = (...args) => {
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, width, height);
    paint(context, ...args);
    map.needsUpdate = true;
  };
  return { map, draw };
}

function trapezoid(bottom, top, height) {
  const THREE = T();
  const shape = new THREE.Shape();
  shape.moveTo(-bottom / 2, 0);
  shape.lineTo(bottom / 2, 0);
  shape.lineTo(top / 2, height);
  shape.lineTo(-top / 2, height);
  shape.lineTo(-bottom / 2, 0);
  return shape;
}

function torsoGeometry() {
  const THREE = T();
  const depth = 8 * MM;
  const geometry = new THREE.ExtrudeGeometry(trapezoid(15.6 * MM, 12 * MM, 12.8 * MM), { depth: depth - 0.06, bevelEnabled: true, bevelThickness: 0.03, bevelSize: 0.03, bevelSegments: 2 });
  geometry.translate(0, 0, -(depth - 0.06) / 2);
  return geometry;
}

function frontDecal(bottom, top, height) {
  const THREE = T();
  const geometry = new THREE.ShapeGeometry(trapezoid(bottom, top, height));
  const position = geometry.attributes.position;
  const uv = geometry.attributes.uv;
  for (let i = 0; i < position.count; i++) uv.setXY(i, (position.getX(i) + bottom / 2) / bottom, position.getY(i) / height);
  return geometry;
}

function headGeometry(radius = 4.9 * MM, height = 9 * MM) {
  const THREE = T();
  const fillet = 1.3 * MM;
  const points = [new THREE.Vector2(0, 0), new THREE.Vector2(radius - fillet, 0)];
  for (let i = 0; i <= 4; i++) {
    const a = -Math.PI / 2 + (i / 4) * (Math.PI / 2);
    points.push(new THREE.Vector2(radius - fillet + Math.cos(a) * fillet, fillet + Math.sin(a) * fillet));
  }
  for (let i = 0; i <= 4; i++) {
    const a = (i / 4) * (Math.PI / 2);
    points.push(new THREE.Vector2(radius - fillet + Math.cos(a) * fillet, height - fillet + Math.sin(a) * fillet));
  }
  points.push(new THREE.Vector2(0, height));
  const geometry = new THREE.LatheGeometry(points, 32, Math.PI, Math.PI * 2);
  return geometry;
}

function handGeometry() {
  const THREE = T();
  const geometry = new THREE.TorusGeometry(2.2 * MM, 1.1 * MM, 8, 16, Math.PI * 1.45);
  geometry.rotateZ(Math.PI * 0.77);
  geometry.rotateY(Math.PI / 2);
  return geometry;
}

function armGeometry(length) {
  const THREE = T();
  const upper = new THREE.CylinderGeometry(2.6 * MM, 2.9 * MM, length * 0.55, 14);
  upper.translate(0, -length * 0.275, 0);
  return upper;
}

const FACES = {
  determined(context, skin) {
    context.fillStyle = skin;
    context.fillRect(0, 0, 256, 128);
    context.fillStyle = "#1c1410";
    context.fillRect(100, 44, 10, 14);
    context.fillRect(146, 44, 10, 14);
    context.fillStyle = "#ffffff";
    context.fillRect(102, 46, 3, 3);
    context.fillRect(148, 46, 3, 3);
    context.strokeStyle = "#5a3820";
    context.lineWidth = 4;
    context.beginPath(); context.moveTo(94, 36); context.lineTo(114, 33); context.moveTo(142, 33); context.lineTo(162, 36); context.stroke();
    context.strokeStyle = "#1c1410";
    context.lineWidth = 4;
    context.beginPath(); context.arc(128, 66, 15, Math.PI * 0.15, Math.PI * 0.75); context.stroke();
    context.fillStyle = "rgba(170, 90, 60, 0.45)";
    [[92, 66], [98, 70], [158, 66], [164, 70], [95, 74]].forEach(([x, y]) => { context.beginPath(); context.arc(x, y, 2, 0, Math.PI * 2); context.fill(); });
  },
  grumpy(context, skin) {
    context.fillStyle = skin;
    context.fillRect(0, 0, 256, 128);
    context.strokeStyle = "#2a1408";
    context.lineWidth = 5;
    context.beginPath(); context.moveTo(92, 34); context.lineTo(116, 44); context.moveTo(164, 34); context.lineTo(140, 44); context.stroke();
    context.lineWidth = 4;
    context.beginPath(); context.arc(128, 86, 9, Math.PI * 1.15, Math.PI * 1.85); context.stroke();
    context.fillStyle = "rgba(90, 40, 20, 0.35)";
    for (let i = 0; i < 18; i++) context.fillRect(96 + (i * 37) % 64, 92 + (i * 13) % 14, 2, 2);
  },
  calm(context, skin) {
    context.fillStyle = skin;
    context.fillRect(0, 0, 256, 128);
    context.fillStyle = "rgba(60, 50, 40, 0.5)";
    context.fillRect(116, 70, 24, 3);
  },
};

const TORSOS = {
  anakin(context) {
    context.fillStyle = "#c9ae80";
    context.fillRect(0, 0, 256, 256);
    context.fillStyle = "#b39566";
    context.beginPath(); context.moveTo(40, 256); context.lineTo(150, 0); context.lineTo(196, 0); context.lineTo(96, 256); context.fill();
    context.strokeStyle = "#8a6c44";
    context.lineWidth = 4;
    context.beginPath(); context.moveTo(40, 256); context.lineTo(150, 0); context.moveTo(96, 256); context.lineTo(196, 0); context.stroke();
    context.fillStyle = "#6b4426";
    context.fillRect(0, 30, 256, 34);
    context.fillStyle = "#d8b85a";
    context.fillRect(116, 34, 24, 26);
    context.fillStyle = "#6b4426";
    context.fillRect(122, 40, 12, 14);
    context.fillStyle = "#7a5030";
    context.fillRect(40, 64, 36, 30);
    context.fillRect(180, 64, 36, 30);
    context.fillStyle = "#c84848";
    context.fillRect(60, 120, 10, 120);
    context.strokeStyle = "#9a7a50";
    context.lineWidth = 3;
    context.beginPath(); context.moveTo(128, 256); context.lineTo(128, 190); context.lineTo(100, 150); context.moveTo(128, 190); context.lineTo(156, 150); context.stroke();
  },
  sebulba(context) {
    context.fillStyle = "#ba7040";
    context.fillRect(0, 0, 256, 256);
    context.fillStyle = "#6e4e36";
    context.beginPath(); context.moveTo(20, 256); context.lineTo(70, 0); context.lineTo(110, 0); context.lineTo(70, 256); context.fill();
    context.beginPath(); context.moveTo(236, 256); context.lineTo(186, 0); context.lineTo(146, 0); context.lineTo(186, 256); context.fill();
    context.fillRect(0, 70, 256, 26);
    context.fillStyle = "#c8c8c8";
    [[56, 74], [184, 74], [86, 170], [158, 170]].forEach(([x, y]) => { context.fillRect(x, y, 18, 18); context.fillStyle = "#6e4e36"; context.fillRect(x + 5, y + 5, 8, 8); context.fillStyle = "#c8c8c8"; });
    context.fillStyle = "#4a3424";
    context.fillRect(0, 0, 256, 20);
  },
  quadinaros(context) {
    context.fillStyle = "#a8a476";
    context.fillRect(0, 0, 256, 256);
    context.strokeStyle = "#7c7854";
    context.lineWidth = 3;
    [60, 196].forEach((x) => { context.beginPath(); context.moveTo(x, 0); context.lineTo(x, 256); context.stroke(); });
    context.beginPath(); context.moveTo(0, 150); context.lineTo(256, 150); context.stroke();
    context.fillStyle = "#cfc690";
    context.fillRect(78, 40, 100, 90);
    context.strokeStyle = "#7c7854";
    context.strokeRect(78, 40, 100, 90);
    context.fillStyle = "#d8c040";
    context.fillRect(90, 54, 22, 10);
    context.fillStyle = "#4a6a9a";
    context.fillRect(140, 54, 26, 10);
    context.fillStyle = "#6a6648";
    context.fillRect(120, 160, 16, 96);
  },
};

const LEGS = {
  anakin(context) {
    context.fillStyle = "#c9ae80";
    context.fillRect(0, 0, 128, 256);
    context.fillStyle = "#5a3a22";
    context.fillRect(0, 150, 128, 106);
    context.fillStyle = "#3a2414";
    context.fillRect(0, 150, 128, 10);
    context.fillStyle = "#b39566";
    context.fillRect(40, 70, 48, 40);
  },
  sebulba(context) {
    context.fillStyle = "#7a5638";
    context.fillRect(0, 0, 128, 256);
    context.strokeStyle = "#5a3e28";
    context.lineWidth = 6;
    for (let y = 60; y < 256; y += 30) { context.beginPath(); context.moveTo(0, y); context.lineTo(128, y + 14); context.stroke(); }
  },
  quadinaros(context) {
    context.fillStyle = "#a8a476";
    context.fillRect(0, 0, 128, 256);
    context.fillStyle = "#cfc690";
    context.fillRect(30, 90, 68, 50);
    context.strokeStyle = "#7c7854";
    context.lineWidth = 3;
    context.strokeRect(30, 90, 68, 50);
    context.fillStyle = "#5a5640";
    context.fillRect(0, 200, 128, 56);
  },
};

const LOOKS = {
  anakin: { skin: "--skinlight3d", torso: "--tunic3d", arm: "--tunic3d", hand: "--leather3d", hip: "--leather3d", leg: "--tunic3d", foot: "--leather3d", face: "determined", short: false, scale: 1.55 },
  sebulba: { skin: "--alien3d", torso: "--alien3d", arm: "--alien3d", hand: "--alien3d", hip: "--vest3d", leg: "--vest3d", foot: "--alien3d", face: "grumpy", short: true, scale: 1.65 },
  quadinaros: { skin: "--alienpale3d", torso: "--suit3d", arm: "--suit3d", hand: "--alienpale3d", hip: "--suit3d", leg: "--suit3d", foot: "--shadow3d", face: "calm", short: false, scale: 1.3 },
};

export function buildMinifigure(kit, id, owner) {
  const THREE = T();
  const look = LOOKS[id] || LOOKS.anakin;
  const group = new THREE.Group();
  const build = builder(group);
  const dark = kit.mat("--shadow3d", { owner });
  const steel = kit.mat("--steel3d", { owner });
  build.brick({ w: 4 * P - 0.01, d: 4 * P - 0.01, h: BRICK - 0.01, material: steel, top: false });
  build.brick({ y: BRICK, w: 4 * P - 0.01, d: 4 * P - 0.01, h: PLATE - 0.01, material: dark, top: false });
  const base = BRICK + PLATE;
  [[-1.5, -1.5], [1.5, -1.5], [-1.5, 1.5], [1.5, 1.5]].forEach(([i, j]) => build.stud(dark, new THREE.Vector3(i * P, base, j * P)));
  build.finish();

  const material = (token) => kit.mat(token, { owner, rough: 0.22 });
  const decal = (map) => kit.raw(new THREE.MeshPhysicalMaterial({ map, transparent: true, roughness: 0.22, clearcoat: 0.5, clearcoatRoughness: 0.15, polygonOffset: true, polygonOffsetFactor: -1 }));
  const skin = material(look.skin);

  const figure = new THREE.Group();
  figure.position.y = base;
  const body = new THREE.Group();
  body.scale.setScalar(look.scale);
  figure.add(body);

  const legLength = look.short ? 6.4 * MM : 12.4 * MM;
  const legTexture = canvasTexture(128, 256, LEGS[id] || LEGS.anakin);
  legTexture.draw();
  const legs = [-1, 1].map((side) => {
    const pivot = new THREE.Group();
    pivot.position.set(side * 3.95 * MM, legLength + 0.4 * MM, 0);
    const leg = mesh(brickGeometry(7.6 * MM, 6.2 * MM, legLength, 0.025), material(look.leg), 0, -legLength / 2, 0);
    leg.geometry = leg.geometry.clone();
    pivot.add(leg);
    const foot = mesh(brickGeometry(7.6 * MM, 9.4 * MM, 2.2 * MM, 0.02), material(look.foot), 0, -legLength + 1.1 * MM - 0.02, 1.4 * MM);
    pivot.add(foot);
    const sole = mesh(brickGeometry(7.2 * MM, 9 * MM, 0.5 * MM, 0.01), dark, 0, -legLength + 0.25 * MM - 0.02, 1.4 * MM);
    pivot.add(sole);
    const print = mesh(new THREE.PlaneGeometry(7.4 * MM, legLength - 2.4 * MM), decal(legTexture.map), 0, -legLength / 2 + 1.2 * MM, 3.12 * MM);
    print.castShadow = false;
    pivot.add(print);
    body.add(pivot);
    return pivot;
  });
  const hipY = legLength + 0.4 * MM;
  body.add(mesh(brickGeometry(15.6 * MM, 6.2 * MM, 3 * MM, 0.02), material(look.hip), 0, hipY + 1.5 * MM, 0));
  const pin = mesh(new THREE.CylinderGeometry(1.6 * MM, 1.6 * MM, 15.8 * MM, 10), dark, 0, hipY + 0.4 * MM, 0);
  pin.rotation.z = Math.PI / 2;
  body.add(pin);

  const torsoY = hipY + 3 * MM;
  const torso = new THREE.Group();
  torso.position.y = torsoY;
  torso.add(mesh(torsoGeometry(), material(look.torso)));
  const torsoTexture = canvasTexture(256, 256, TORSOS[id] || TORSOS.anakin);
  torsoTexture.draw();
  const front = mesh(frontDecal(15.4 * MM, 11.8 * MM, 12.6 * MM), decal(torsoTexture.map), 0, 0.1 * MM, 4.02 * MM);
  front.castShadow = false;
  torso.add(front);
  body.add(torso);

  const shoulderY = 10.6 * MM;
  const arms = [-1, 1].map((side) => {
    const pivot = new THREE.Group();
    pivot.position.set(side * 6.9 * MM, shoulderY, 0);
    pivot.rotation.z = side * 0.12;
    const shoulder = mesh(new THREE.SphereGeometry(3.2 * MM, 14, 10), material(look.arm));
    pivot.add(shoulder);
    const upper = mesh(armGeometry(11 * MM), material(look.arm), side * 0.6 * MM, 0, 0);
    upper.rotation.z = side * 0.18;
    pivot.add(upper);
    const elbow = new THREE.Group();
    elbow.position.set(side * 1.4 * MM, -5.6 * MM, 0);
    elbow.rotation.x = -0.55;
    elbow.add(mesh(new THREE.SphereGeometry(2.8 * MM, 12, 8), material(look.arm)));
    const fore = mesh(armGeometry(10 * MM), material(look.arm));
    elbow.add(fore);
    const wrist = new THREE.Group();
    wrist.position.y = -5.4 * MM;
    wrist.add(mesh(new THREE.CylinderGeometry(1.5 * MM, 1.5 * MM, 2.4 * MM, 10), material(look.hand), 0, -0.9 * MM, 0));
    const hand = mesh(handGeometry(), material(look.hand), 0, -3.6 * MM, 0.4 * MM);
    wrist.add(hand);
    if (id === "sebulba") {
      for (let k = 0; k < 3; k++) wrist.add(flat(mesh(new THREE.TorusGeometry(2.4 * MM, 0.5 * MM, 6, 14), material("--vest3d"), 0, 0.6 * MM + k * 1.1 * MM, 0)));
    }
    elbow.add(wrist);
    pivot.add(elbow);
    torso.add(pivot);
    return { pivot, elbow, wrist };
  });

  const neck = mesh(new THREE.CylinderGeometry(2.6 * MM, 2.6 * MM, 1.8 * MM, 14), skin, 0, 12.8 * MM + 0.9 * MM, 0);
  torso.add(neck);
  const head = new THREE.Group();
  head.position.y = 12.8 * MM + 1.8 * MM;
  const faceTexture = canvasTexture(256, 128, (context) => FACES[look.face](context, skin.color.clone().convertLinearToSRGB().getStyle()));
  faceTexture.draw();
  kit.onTheme(() => faceTexture.draw());
  const faceMaterial = kit.raw(new THREE.MeshPhysicalMaterial({ map: faceTexture.map, roughness: 0.25, clearcoat: 0.4 }));
  const skull = mesh(headGeometry(), faceMaterial);
  head.add(skull);
  head.add(mesh(new THREE.CylinderGeometry(2.6 * MM, 2.6 * MM, 1.7 * MM, 14, 1, true), skin, 0, 9 * MM + 0.85 * MM, 0));
  torso.add(head);

  const glass = kit.trans("--visor3d", { owner, opacity: 0.75 });
  const glare = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8 });
  const goggles = (radius, y, z, spread, rim) => {
    const pair = new THREE.Group();
    for (const side of [-1, 1]) {
      const ring = mesh(new THREE.TorusGeometry(radius, radius * 0.28, 8, 20), rim, side * spread, y, z);
      const lens = mesh(new THREE.CircleGeometry(radius * 0.95, 20), glass, side * spread, y, z + 0.005);
      const shine = mesh(new THREE.CircleGeometry(radius * 0.25, 10), glare, side * spread - radius * 0.35, y + radius * 0.35, z + 0.01);
      shine.castShadow = false;
      pair.add(ring, lens, shine);
    }
    return pair;
  };

  if (id === "anakin") {
    const leather = material("--leather3d");
    const cap = mesh(new THREE.SphereGeometry(5.3 * MM, 24, 12, 0, Math.PI * 2, 0, Math.PI * 0.5), leather, 0, 6.6 * MM, -0.2 * MM);
    cap.scale.set(1, 0.95, 1.05);
    head.add(cap);
    for (const side of [-1, 1]) {
      const flap = mesh(brickGeometry(1.2 * MM, 4.6 * MM, 5.4 * MM, 0.01), leather, side * 5.2 * MM, 3.4 * MM, -0.4 * MM);
      head.add(flap);
    }
    head.add(flat(mesh(new THREE.TorusGeometry(5.3 * MM, 0.6 * MM, 6, 28), leather, 0, 6.9 * MM, 0)));
    const pair = goggles(1.9 * MM, 9.6 * MM, 4.2 * MM, 2.4 * MM, material("--steel3d"));
    pair.rotation.x = -0.35;
    head.add(pair);
  } else if (id === "sebulba") {
    const cover = mesh(new THREE.SphereGeometry(5.4 * MM, 28, 16, 0, Math.PI * 2, 0, Math.PI * 0.62), skin, 0, 5.6 * MM, -0.6 * MM);
    cover.scale.set(1.75, 0.95, 1.2);
    head.add(cover);
    for (const side of [-1, 1]) {
      const ear = mesh(new THREE.ConeGeometry(1.2 * MM, 3.6 * MM, 8), skin, side * 10.2 * MM, 6.8 * MM, -0.6 * MM);
      ear.rotation.z = -side * 1.2;
      head.add(ear);
    }
    const strap = flat(mesh(new THREE.TorusGeometry(5.6 * MM, 0.6 * MM, 6, 28), material("--vest3d"), 0, 6.4 * MM, -0.6 * MM));
    strap.scale.set(1.62, 1.12, 1);
    head.add(strap);
    head.add(goggles(2.6 * MM, 6.6 * MM, 6.4 * MM, 3.2 * MM, material("--yellow-light")));
  } else {
    const suit = material("--suit3d");
    const helmet = new THREE.Group();
    helmet.position.y = 7.8 * MM;
    helmet.add(mesh(new THREE.CylinderGeometry(5.1 * MM, 5.1 * MM, 2.6 * MM, 24), suit));
    const neckSegments = 5;
    for (let k = 0; k < neckSegments; k++) {
      const y = 2 * MM + k * 3 * MM;
      helmet.add(mesh(new THREE.CylinderGeometry(1.9 * MM, 2.2 * MM, 2.6 * MM, 12), skin, 0, y, 0));
      helmet.add(flat(mesh(new THREE.TorusGeometry(2.1 * MM, 0.4 * MM, 6, 14), material("--suit3d"), 0, y - 1.3 * MM, 0)));
    }
    const dome = mesh(new THREE.SphereGeometry(5.6 * MM, 24, 16), skin, 0, 2 * MM + neckSegments * 3 * MM + 3 * MM, 0.6 * MM);
    dome.scale.set(1, 0.92, 1.02);
    helmet.add(dome);
    for (const side of [-1, 1]) {
      const eye = mesh(new THREE.SphereGeometry(1.5 * MM, 14, 10), material("--eyes3d"), side * 2.3 * MM, 2 * MM + neckSegments * 3 * MM + 3.2 * MM, 5.2 * MM);
      eye.scale.set(1, 1.25, 0.6);
      helmet.add(eye);
      const spark = mesh(new THREE.SphereGeometry(0.4 * MM, 8, 6), glare, side * 2.3 * MM - 0.5 * MM, 2 * MM + neckSegments * 3 * MM + 3.9 * MM, 6.0 * MM);
      spark.castShadow = false;
      helmet.add(spark);
    }
    head.add(helmet);
    torso.add(mesh(brickGeometry(9 * MM, 1.2 * MM, 6 * MM, 0.01), material("--alienpale3d"), 0, 8.4 * MM, 4.4 * MM));
    for (const side of [-1, 1]) legs[side < 0 ? 0 : 1].add(mesh(brickGeometry(6.2 * MM, 1 * MM, 3.4 * MM, 0.01), material("--alienpale3d"), 0, -legLength * 0.45, 3.4 * MM));
  }

  const moving = new Set();
  [...legs, ...arms.map((arm) => arm.pivot), head].forEach((part) => part.traverse((child) => moving.add(child)));
  bake(torso, moving);
  bake(body, new Set([torso, ...legs].flatMap((part) => { const list = []; part.traverse((child) => list.push(child)); return list; })));
  group.add(figure);

  return {
    group,
    figure,
    base,
    wobble: id === "quadinaros",
    arms: arms.map((arm) => arm.pivot),
    rig: { body, torso, head, legs, arms, look, id, hip: (legLength + 0.4 * MM) * look.scale },
  };
}

export function pose(rig, state) {
  const { torso, head, legs, arms, id } = rig;
  const { clock, selected, hovered, cheering, walking, seated, reduced, lift } = state;
  const breathe = reduced ? 0 : Math.sin(clock * 2.2) * 0.012;
  torso.scale.set(1 + breathe * 0.5, 1 + breathe, 1 + breathe * 0.5);
  head.rotation.y = reduced ? 0 : hovered ? Math.sin(clock * 2) * 0.5 : Math.sin(clock * 0.7 + id.length) * 0.25 * (Math.sin(clock * 0.31) > 0.4 ? 1 : 0);
  legs.forEach((leg, i) => { leg.rotation.x = seated ? -Math.PI / 2 : walking ? Math.sin(clock * 12 + i * Math.PI) * 0.6 : 0; });
  const [left, right] = arms;
  [left, right].forEach((arm, i) => {
    arm.pivot.rotation.x = walking ? -Math.sin(clock * 12 + i * Math.PI) * 0.5 : Math.sin(clock * 1.1 + i) * 0.04;
    arm.wrist.rotation.y = 0;
    arm.elbow.rotation.x = -0.55;
  });
  if (seated) {
    left.pivot.rotation.x = right.pivot.rotation.x = -1.05;
    left.elbow.rotation.x = right.elbow.rotation.x = -0.3;
    head.rotation.y = reduced ? 0 : Math.sin(clock * 0.9) * 0.2;
    return;
  }
  if (cheering) {
    left.pivot.rotation.x = -2.9 + Math.sin(clock * 12) * 0.2;
    right.pivot.rotation.x = -2.9 - Math.sin(clock * 12) * 0.2;
    left.elbow.rotation.x = right.elbow.rotation.x = -0.2;
    return;
  }
  if (!selected || reduced || walking) return;
  const amount = Math.min(1, Math.max(0, lift));
  if (id === "anakin") {
    right.pivot.rotation.x = -1.35 * amount;
    right.elbow.rotation.x = -1.2 * amount;
    right.wrist.rotation.y = 1.4 * amount;
  } else if (id === "sebulba") {
    const pump = Math.max(0, Math.sin(clock * 6));
    right.pivot.rotation.x = (-2.2 - pump * 0.5) * amount;
    right.elbow.rotation.x = (-1.4 + pump * 0.9) * amount;
  } else {
    right.pivot.rotation.x = -2.6 * amount;
    right.elbow.rotation.x = -0.35 * amount;
    right.wrist.rotation.y = Math.sin(clock * 9) * 0.9 * amount;
    left.pivot.rotation.x = Math.sin(clock * 7) * 0.12;
  }
}
