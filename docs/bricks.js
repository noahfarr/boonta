export const P = 0.8;
export const BRICK = 0.96;
export const PLATE = 0.32;
export const STUD = { radius: 0.24, height: 0.17 };

const T = () => window.THREE;
const geometries = new Map();

export function brickGeometry(w, d, h, bevel = 0.045) {
  const key = `${w.toFixed(3)}:${d.toFixed(3)}:${h.toFixed(3)}:${bevel}`;
  if (geometries.has(key)) return geometries.get(key);
  const THREE = T();
  const b = Math.min(bevel, w / 4, d / 4, h / 4);
  const shape = new THREE.Shape();
  shape.moveTo(-w / 2 + b, -d / 2 + b);
  shape.lineTo(w / 2 - b, -d / 2 + b);
  shape.lineTo(w / 2 - b, d / 2 - b);
  shape.lineTo(-w / 2 + b, d / 2 - b);
  shape.lineTo(-w / 2 + b, -d / 2 + b);
  const geometry = new THREE.ExtrudeGeometry(shape, { depth: Math.max(0.001, h - 2 * b), bevelEnabled: true, bevelThickness: b, bevelSize: b, bevelSegments: 1, steps: 1, curveSegments: 1 });
  geometry.rotateX(-Math.PI / 2);
  geometry.translate(0, b - h / 2, 0);
  geometry.computeVertexNormals();
  geometries.set(key, geometry);
  return geometry;
}

function merge(parts) {
  const THREE = T();
  const names = ["position", "normal", "uv"];
  const total = parts.reduce((sum, geometry) => sum + geometry.attributes.position.count, 0);
  const merged = new THREE.BufferGeometry();
  names.forEach((name) => {
    const size = name === "uv" ? 2 : 3;
    const array = new Float32Array(total * size);
    let offset = 0;
    parts.forEach((geometry) => {
      const attribute = geometry.attributes[name];
      if (attribute) array.set(attribute.array.subarray(0, attribute.count * size), offset);
      offset += geometry.attributes.position.count * size;
    });
    merged.setAttribute(name, new THREE.BufferAttribute(array, size));
  });
  merged.computeBoundingSphere();
  return merged;
}

export function studGeometry() {
  if (!geometries.has("stud")) {
    const THREE = T();
    const side = new THREE.CylinderGeometry(STUD.radius, STUD.radius, STUD.height, 10, 1, true).toNonIndexed();
    side.translate(0, STUD.height / 2, 0);
    const top = new THREE.CircleGeometry(STUD.radius, 10).toNonIndexed();
    top.rotateX(-Math.PI / 2);
    top.translate(0, STUD.height, 0);
    geometries.set("stud", merge([side, top]));
  }
  return geometries.get("stud");
}

export function studMaps(renderer) {
  const THREE = T();
  const size = 64;
  const color = document.createElement("canvas");
  color.width = color.height = size;
  const paint = color.getContext("2d");
  paint.fillStyle = "#ffffff";
  paint.fillRect(0, 0, size, size);
  paint.fillStyle = "rgba(0, 0, 0, 0.16)";
  paint.beginPath(); paint.arc(35, 36, 19, 0, Math.PI * 2); paint.fill();
  paint.fillStyle = "#f2f2f2";
  paint.beginPath(); paint.arc(32, 32, 19, 0, Math.PI * 2); paint.fill();
  paint.strokeStyle = "rgba(255, 255, 255, 0.95)";
  paint.lineWidth = 3;
  paint.beginPath(); paint.arc(32, 32, 16, Math.PI * 0.95, Math.PI * 1.6); paint.stroke();
  paint.strokeStyle = "rgba(0, 0, 0, 0.12)";
  paint.beginPath(); paint.arc(32, 32, 18, Math.PI * 0.05, Math.PI * 0.7); paint.stroke();
  const bump = document.createElement("canvas");
  bump.width = bump.height = size;
  const lift = bump.getContext("2d");
  lift.fillStyle = "#000";
  lift.fillRect(0, 0, size, size);
  lift.fillStyle = "#fff";
  lift.beginPath(); lift.arc(32, 32, 19, 0, Math.PI * 2); lift.fill();
  const anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
  const make = (source) => {
    const map = new THREE.CanvasTexture(source);
    map.wrapS = map.wrapT = THREE.RepeatWrapping;
    map.anisotropy = anisotropy;
    return map;
  };
  const base = { map: make(color), bump: make(bump) };
  const sized = (repeatX, repeatY) => {
    const map = base.map.clone();
    const bumpMap = base.bump.clone();
    map.repeat.set(repeatX, repeatY);
    bumpMap.repeat.set(repeatX, repeatY);
    map.needsUpdate = true;
    bumpMap.needsUpdate = true;
    return { map, bumpMap };
  };
  return { world: sized(1 / P, 1 / P), sized };
}

export function builder(root) {
  const THREE = T();
  const studs = new Map();
  const matrix = new THREE.Matrix4();
  const quaternion = new THREE.Quaternion();
  const unit = new THREE.Vector3(1, 1, 1);
  const addStud = (material, position, rotation) => {
    if (!studs.has(material)) studs.set(material, []);
    quaternion.setFromEuler(rotation || new THREE.Euler());
    studs.get(material).push(new THREE.Matrix4().compose(position, quaternion, unit));
  };
  const pending = new Map();
  const api = {
    brick({ x = 0, y = 0, z = 0, w, d, h = BRICK, material, top = true, yaw = 0, parent = root }) {
      const cos = Math.abs(Math.cos(yaw)), sin = Math.abs(Math.sin(yaw));
      const hx = (w * cos + d * sin) / 2, hz = (w * sin + d * cos) / 2;
      if (!root.userData.boxes) root.userData.boxes = [];
      root.userData.boxes.push(new THREE.Box3(new THREE.Vector3(x - hx, y, z - hz), new THREE.Vector3(x + hx, y + h, z + hz)));
      const geometry = brickGeometry(w, d, h).clone();
      geometry.rotateY(yaw);
      geometry.translate(x, y + h / 2, z);
      if (!pending.has(material)) pending.set(material, []);
      pending.get(material).push(geometry);
      if (top && parent === root) {
        const columns = Math.max(1, Math.round(w / P)), rows = Math.max(1, Math.round(d / P));
        const cos = Math.cos(yaw), sin = Math.sin(yaw);
        for (let i = 0; i < columns; i++) for (let j = 0; j < rows; j++) {
          const lx = (i - (columns - 1) / 2) * P, lz = (j - (rows - 1) / 2) * P;
          addStud(material, new THREE.Vector3(x + lx * cos + lz * sin, y + h, z - lx * sin + lz * cos));
        }
      }
    },
    stud(material, position, rotation) { addStud(material, position, rotation); },
    finish() {
      pending.forEach((parts, material) => {
        const mesh = new THREE.Mesh(merge(parts), material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.userData.merged = true;
        root.add(mesh);
      });
      pending.clear();
      studs.forEach((list, material) => {
        const mesh = new THREE.InstancedMesh(studGeometry(), material, list.length);
        list.forEach((m, i) => mesh.setMatrixAt(i, m));
        mesh.castShadow = true;
        mesh.userData.studs = true;
        root.add(mesh);
      });
      studs.clear();
    },
  };
  return api;
}

export function faceTexture(skin, mood = "smile") {
  const THREE = T();
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 64;
  const map = new THREE.CanvasTexture(canvas);
  const draw = (color) => {
    const paint = canvas.getContext("2d");
    paint.fillStyle = color;
    paint.fillRect(0, 0, 128, 64);
    paint.fillStyle = "#1a1a1a";
    paint.strokeStyle = "#1a1a1a";
    paint.lineWidth = 3;
    if (mood !== "none") {
      paint.fillRect(55, 24, 5, 8);
      paint.fillRect(68, 24, 5, 8);
    }
    paint.beginPath();
    if (mood === "smile") paint.arc(64, 34, 9, Math.PI * 0.2, Math.PI * 0.8);
    if (mood === "determined") {
      paint.moveTo(52, 20); paint.lineTo(61, 22);
      paint.moveTo(76, 20); paint.lineTo(67, 22);
      paint.moveTo(58, 42); paint.lineTo(70, 41);
    }
    if (mood === "grumpy") paint.arc(64, 50, 8, Math.PI * 1.15, Math.PI * 1.85);
    paint.stroke();
    map.needsUpdate = true;
  };
  draw(skin);
  return { map, draw };
}

export function grilleTexture() {
  const THREE = T();
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 32;
  const paint = canvas.getContext("2d");
  paint.fillStyle = "#cfcfcf";
  paint.fillRect(0, 0, 32, 32);
  paint.fillStyle = "#3a3a3a";
  for (let y = 3; y < 32; y += 6) paint.fillRect(0, y, 32, 3);
  const map = new THREE.CanvasTexture(canvas);
  return map;
}

export function bake(group, keep = new Set()) {
  const THREE = T();
  group.updateMatrixWorld(true);
  const inverse = new THREE.Matrix4().copy(group.matrixWorld).invert();
  const buckets = new Map();
  const removed = [];
  group.traverse((child) => {
    if (!child.isMesh || child.isInstancedMesh || keep.has(child)) return;
    const relative = new THREE.Matrix4().multiplyMatrices(inverse, child.matrixWorld);
    if (relative.determinant() < 0) return;
    const geometry = child.geometry.index ? child.geometry.toNonIndexed() : child.geometry.clone();
    geometry.applyMatrix4(relative);
    if (!buckets.has(child.material)) buckets.set(child.material, { parts: [], cast: false });
    const bucket = buckets.get(child.material);
    bucket.parts.push(geometry);
    bucket.cast = bucket.cast || child.castShadow;
    removed.push(child);
  });
  removed.forEach((child) => child.parent.remove(child));
  buckets.forEach(({ parts, cast }, material) => {
    const mesh = new THREE.Mesh(merge(parts), material);
    mesh.userData.baked = true;
    mesh.castShadow = cast;
    mesh.receiveShadow = true;
    group.add(mesh);
  });
}
