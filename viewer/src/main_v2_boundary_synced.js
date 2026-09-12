import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import "./style.css";


const RUN_COLORS = {
  A: 0x174f8a,
  B: 0xb54a26,
};

const NEWEST_TOKEN_COLOR = 0x000000;

const BIRD_LENGTH = 0.15;
const BIRD_WIDTH = 0.07;

const PLAYBACK_SPEED = 10.0;

const GLOW_VIEW_SECONDS = 2.4;
const GLOW_SIZE = BIRD_WIDTH * 5.5;

const CAMERA_RADIUS_MULTIPLIER = 3.0;
const MIN_CAMERA_DISTANCE = 4.5;

const BINARY_MAGIC = "LLMCHS01";
const BINARY_VERSION = 1;

const HEADER_BYTES = 24;
const INDEX_ENTRY_BYTES = 24;
const FLOATS_PER_BIRD = 6;


const app = document.querySelector("#app");

const viewerEnhancementStyle = document.createElement("style");
viewerEnhancementStyle.textContent = `
  .scene {
    position: relative;
  }

  .newest-token-box {
    position: absolute;
    right: 12px;
    bottom: 12px;
    z-index: 10;
    display: flex;
    align-items: baseline;
    gap: 7px;
    max-width: calc(100% - 24px);
    padding: 5px 8px;
    border: 1px solid rgba(0, 0, 0, 0.18);
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.90);
    color: #111;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
    line-height: 1.2;
    pointer-events: none;
    backdrop-filter: blur(3px);
  }

  .newest-token-label {
    color: #777;
    font-size: 9px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  .newest-token-value {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: pre;
  }

  .generation-block {
    margin-top: 14px;
    padding: 14px 2px 4px;
    border-top: 1px solid rgba(0, 0, 0, 0.12);
  }

  .generation-heading {
    margin-bottom: 8px;
    color: #555;
    font-size: 11px;
    font-weight: 650;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  .generation-text {
    height: 180px;
    overflow-y: auto;
    overscroll-behavior: contain;
    padding-right: 10px;
    color: #171717;
    font-family: Georgia, "Times New Roman", serif;
    font-size: 15px;
    line-height: 1.62;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }

  .generation-token {
    transition:
      color 90ms linear,
      background-color 90ms linear;
  }

  #generation-a .generation-token.current-token {
    color: #174f8a;
    background: rgba(23, 79, 138, 0.14);
    border-radius: 2px;
  }

  #generation-b .generation-token.current-token {
    color: #b54a26;
    background: rgba(181, 74, 38, 0.14);
    border-radius: 2px;
  }
`;
document.head.appendChild(viewerEnhancementStyle);

app.innerHTML = `
  <header class="topbar">
    <div>
      <div class="eyebrow">LLM CHAOS</div>
      <h1>Trajectory divergence</h1>
    </div>

    <div class="controls">
      <button id="rewind" type="button">↶ Rewind</button>
      <button id="play-pause" type="button">Pause</button>
    </div>
  </header>

  <main class="comparison">
    <section class="panel">
      <div class="panel-header">
        <span class="run-dot run-a"></span>
        <strong>A</strong>
        <span>base prompt</span>
      </div>

      <div id="scene-a" class="scene">
        <div id="newest-a" class="newest-token-box">
          <span class="newest-token-label">newest</span>
          <span class="newest-token-value">—</span>
        </div>
      </div>

      <div class="generation-block">
        <div class="generation-heading">Full generation</div>
        <div id="generation-a" class="generation-text"></div>
      </div>
    </section>

    <section class="panel">
      <div class="panel-header">
        <span class="run-dot run-b"></span>
        <strong>B</strong>
        <span>+ trailing space</span>
      </div>

      <div id="scene-b" class="scene">
        <div id="newest-b" class="newest-token-box">
          <span class="newest-token-label">newest</span>
          <span class="newest-token-value">—</span>
        </div>
      </div>

      <div class="generation-block">
        <div class="generation-heading">Full generation</div>
        <div id="generation-b" class="generation-text"></div>
      </div>
    </section>
  </main>

  <footer class="footer">
    <span id="status">loading compact trajectory…</span>
    <span id="time-display">0.0 s</span>
  </footer>

  <div id="tooltip" class="tooltip">
    <div id="tooltip-token"></div>
    <small id="tooltip-index"></small>
  </div>
`;


const playPauseButton = document.querySelector("#play-pause");
const rewindButton = document.querySelector("#rewind");
const statusDisplay = document.querySelector("#status");
const timeDisplay = document.querySelector("#time-display");
const tooltip = document.querySelector("#tooltip");
const tooltipToken = document.querySelector("#tooltip-token");
const tooltipIndex = document.querySelector("#tooltip-index");


// ============================================================
// BINARY LOADER
// ============================================================

function readAscii(bytes, start, length) {
  let result = "";

  for (let i = 0; i < length; i++) {
    result += String.fromCharCode(bytes[start + i]);
  }

  return result;
}


function parseMotionBinary(buffer, meta) {
  const bytes = new Uint8Array(buffer);
  const magic = readAscii(bytes, 0, 8);

  if (magic !== BINARY_MAGIC) {
    throw new Error(`Unexpected trajectory binary magic: ${magic}`);
  }

  const view = new DataView(buffer);
  const version = view.getUint32(8, true);

  if (version !== BINARY_VERSION) {
    throw new Error(`Unsupported trajectory binary version: ${version}`);
  }

  const frameCount = view.getUint32(12, true);
  const maxBirds = view.getUint32(16, true);

  const frames = new Array(frameCount);
  let indexOffset = HEADER_BYTES;

  for (let i = 0; i < frameCount; i++) {
    const time = view.getFloat64(indexOffset, true);
    const activeCount = view.getUint32(indexOffset + 8, true);
    const birthIndex = view.getInt32(indexOffset + 12, true);
    const floatOffset = view.getUint32(indexOffset + 16, true);

    frames[i] = {
      time,
      active_count: activeCount,
      birth_index: birthIndex < 0 ? null : birthIndex,
      float_offset: floatOffset,
    };

    indexOffset += INDEX_ENTRY_BYTES;
  }

  const payloadByteOffset = HEADER_BYTES + frameCount * INDEX_ENTRY_BYTES;

  if (payloadByteOffset % 4 !== 0) {
    throw new Error("Trajectory payload is not Float32 aligned.");
  }

  const payload = new Float32Array(buffer, payloadByteOffset);

  return {
    ...meta,
    frame_count: frameCount,
    max_birds: maxBirds,
    frames,
    payload,
    _buffer: buffer,
  };
}


async function loadBinaryRun(metaUrl, binaryUrl) {
  const [metaResponse, binaryResponse] = await Promise.all([
    fetch(metaUrl, { cache: "no-store" }),
    fetch(binaryUrl, { cache: "no-store" }),
  ]);

  if (!metaResponse.ok) {
    throw new Error(`Could not load ${metaUrl}: ${metaResponse.status}`);
  }

  if (!binaryResponse.ok) {
    throw new Error(`Could not load ${binaryUrl}: ${binaryResponse.status}`);
  }

  const [meta, buffer] = await Promise.all([
    metaResponse.json(),
    binaryResponse.arrayBuffer(),
  ]);

  return parseMotionBinary(buffer, meta);
}


async function loadData() {
  const [A, B] = await Promise.all([
    loadBinaryRun(
      "/data/run_a_motion_v2.meta.json",
      "/data/run_a_motion_v2.bin"
    ),
    loadBinaryRun(
      "/data/run_b_motion_v2.meta.json",
      "/data/run_b_motion_v2.bin"
    ),
  ]);

  return { A, B };
}


// ============================================================
// FRAME DATA ACCESS
// ============================================================

function birdFloatBase(frame, birdIndex) {
  return frame.float_offset + birdIndex * FLOATS_PER_BIRD;
}


// ============================================================
// BIRD GEOMETRY
// ============================================================

function createBirdGeometry() {
  const geometry = new THREE.BufferGeometry();

  const apex = new THREE.Vector3(0, BIRD_LENGTH, 0);
  const baseY = -BIRD_LENGTH * 0.42;
  const r = BIRD_WIDTH;

  const a = new THREE.Vector3(0, baseY, r);
  const b = new THREE.Vector3(-Math.sqrt(3) * r / 2, baseY, -r / 2);
  const c = new THREE.Vector3(Math.sqrt(3) * r / 2, baseY, -r / 2);

  const vertices = new Float32Array([
    a.x, a.y, a.z,
    c.x, c.y, c.z,
    b.x, b.y, b.z,

    apex.x, apex.y, apex.z,
    a.x, a.y, a.z,
    b.x, b.y, b.z,

    apex.x, apex.y, apex.z,
    b.x, b.y, b.z,
    c.x, c.y, c.z,

    apex.x, apex.y, apex.z,
    c.x, c.y, c.z,
    a.x, a.y, a.z,
  ]);

  geometry.setAttribute(
    "position",
    new THREE.BufferAttribute(vertices, 3)
  );

  geometry.computeVertexNormals();
  return geometry;
}


// ============================================================
// GLOW TEXTURE
// ============================================================

function createGlowTexture(colorHex) {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 128;

  const ctx = canvas.getContext("2d");
  const color = new THREE.Color(colorHex);

  const r = Math.round(color.r * 255);
  const g = Math.round(color.g * 255);
  const b = Math.round(color.b * 255);

  const gradient = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);

  gradient.addColorStop(0.00, `rgba(${r},${g},${b},0.56)`);
  gradient.addColorStop(0.20, `rgba(${r},${g},${b},0.26)`);
  gradient.addColorStop(0.50, `rgba(${r},${g},${b},0.10)`);
  gradient.addColorStop(1.00, `rgba(${r},${g},${b},0.0)`);

  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 128, 128);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;

  return texture;
}


// ============================================================
// FRAME LOOKUP
// ============================================================

function findFramePair(frames, time) {
  if (time <= frames[0].time) {
    return { a: 0, b: 0, t: 0 };
  }

  const last = frames.length - 1;

  if (time >= frames[last].time) {
    return { a: last, b: last, t: 0 };
  }

  let low = 0;
  let high = last;

  while (low + 1 < high) {
    const mid = Math.floor((low + high) / 2);

    if (frames[mid].time <= time) {
      low = mid;
    } else {
      high = mid;
    }
  }

  const start = frames[low];
  const end = frames[high];
  const duration = end.time - start.time;
  const t = duration > 0 ? (time - start.time) / duration : 0;

  return { a: low, b: high, t };
}


// ============================================================
// TOKEN DISPLAY
// ============================================================

function readableToken(token) {
  if (token === null || token === undefined) {
    return "";
  }

  return String(token)
    .replaceAll("\n", "↵")
    .replaceAll("\t", "⇥")
    .replaceAll(" ", "␠");
}


// ============================================================
// CAMERA BOUNDS
// ============================================================

const BODY_PERCENTILE = 0.92;
const NEWEST_TOKEN_WEIGHT = 4.0;

function calculateFlockBounds(birdMeshes, activeCount) {
  if (activeCount <= 0) {
    return {
      centroid: new THREE.Vector3(),
      radius: 1.0,
    };
  }

  const newestIndex = activeCount - 1;

  if (activeCount === 1) {
    return {
      centroid: birdMeshes[0].position.clone(),
      radius: 0.5,
    };
  }

  // Estimate the main body using older birds only.
  const roughBodyCenter = new THREE.Vector3();

  for (let i = 0; i < newestIndex; i++) {
    roughBodyCenter.add(birdMeshes[i].position);
  }

  roughBodyCenter.divideScalar(newestIndex);

  const bodyDistances = [];

  for (let i = 0; i < newestIndex; i++) {
    bodyDistances.push({
      index: i,
      distance: birdMeshes[i].position.distanceTo(roughBodyCenter),
    });
  }

  bodyDistances.sort((a, b) => a.distance - b.distance);

  const keepCount = Math.max(
    1,
    Math.ceil(bodyDistances.length * BODY_PERCENTILE)
  );

  const bodyMembers = bodyDistances.slice(0, keepCount);

  const bodyCenter = new THREE.Vector3();

  for (const member of bodyMembers) {
    bodyCenter.add(birdMeshes[member.index].position);
  }

  bodyCenter.divideScalar(bodyMembers.length);

  const newestPosition = birdMeshes[newestIndex].position;
  const weightedCenter = bodyCenter.clone().multiplyScalar(bodyMembers.length);

  weightedCenter.addScaledVector(
    newestPosition,
    NEWEST_TOKEN_WEIGHT
  );

  weightedCenter.divideScalar(
    bodyMembers.length + NEWEST_TOKEN_WEIGHT
  );

  let bodyRadius = 0;

  for (const member of bodyMembers) {
    bodyRadius = Math.max(
      bodyRadius,
      birdMeshes[member.index].position.distanceTo(weightedCenter)
    );
  }

  const newestRadius = newestPosition.distanceTo(weightedCenter);

  return {
    centroid: weightedCenter,
    radius: Math.max(bodyRadius, newestRadius, 0.5),
  };
}


// ============================================================
// PANEL
// ============================================================

class FlockPanel {
  constructor(container, runName, data) {
    this.container = container;
    this.runName = runName;
    this.data = data;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xffffff);

    this.camera = new THREE.PerspectiveCamera(42, 1, 0.01, 1000);

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
    });

    this.renderer.setClearColor(0xffffff, 1);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(
      this.camera,
      this.renderer.domElement
    );

    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.enablePan = false;

    this.group = new THREE.Group();
    this.scene.add(this.group);

    this.birdMeshes = [];
    this.glows = [];

    this.pointer = new THREE.Vector2(100, 100);
    this.raycaster = new THREE.Raycaster();
    this.hovered = null;
    this.pointerX = 0;
    this.pointerY = 0;
    this.pointerInside = false;

    // Shared materials: same look, much less Three.js object overhead.
    this.normalMaterial = new THREE.MeshStandardMaterial({
      color: RUN_COLORS[this.runName],
      roughness: 0.58,
      metalness: 0.0,
      flatShading: true,
      side: THREE.DoubleSide,
    });

    this.newestMaterial = new THREE.MeshStandardMaterial({
      color: NEWEST_TOKEN_COLOR,
      roughness: 0.58,
      metalness: 0.0,
      flatShading: true,
      side: THREE.DoubleSide,
    });

    this.createLights();
    this.createBirds();
    this.initializeCamera();
    this.bindPointer();
    this.resize();
  }


  createLights() {
    const hemi = new THREE.HemisphereLight(0xffffff, 0xd8dde4, 2.4);
    this.scene.add(hemi);

    const key = new THREE.DirectionalLight(0xffffff, 2.1);
    key.position.set(5, 8, 6);
    this.scene.add(key);
  }


  createBirds() {
    const birdGeometry = createBirdGeometry();
    const glowTexture = createGlowTexture(RUN_COLORS[this.runName]);

    for (const bird of this.data.birds) {
      const mesh = new THREE.Mesh(
        birdGeometry,
        this.normalMaterial
      );

      mesh.visible = false;
      mesh.userData = { bird };
      this.group.add(mesh);
      this.birdMeshes.push(mesh);

      const glowMaterial = new THREE.SpriteMaterial({
        map: glowTexture,
        transparent: true,
        opacity: 0,
        depthWrite: false,
        depthTest: true,
        blending: THREE.NormalBlending,
      });

      const glow = new THREE.Sprite(glowMaterial);
      glow.visible = false;
      glow.scale.set(GLOW_SIZE, GLOW_SIZE, 1);

      this.group.add(glow);
      this.glows.push(glow);
    }
  }


  initializeCamera() {
    this.controls.target.set(0, 0, 0);
    this.camera.position.set(5, 3.5, 8);
    this.controls.update();
  }


  update(simulationTime) {
    const pair = findFramePair(this.data.frames, simulationTime);
    const frameA = this.data.frames[pair.a];
    const frameB = this.data.frames[pair.b];

    const activeCount = frameA.active_count;
    const newestIndex = activeCount - 1;
    const payload = this.data.payload;

    // Reuse these vectors for every bird to avoid per-frame garbage.
    const forward = new THREE.Vector3(0, 1, 0);
    const direction = new THREE.Vector3();

    for (let i = 0; i < this.birdMeshes.length; i++) {
      const mesh = this.birdMeshes[i];

      if (i >= activeCount) {
        mesh.visible = false;
        this.glows[i].visible = false;
        continue;
      }

      mesh.visible = true;
      mesh.material =
        i === newestIndex
          ? this.newestMaterial
          : this.normalMaterial;

      const baseA = birdFloatBase(frameA, i);
      const canInterpolate = i < frameB.active_count;
      const baseB = canInterpolate
        ? birdFloatBase(frameB, i)
        : baseA;

      const px = THREE.MathUtils.lerp(
        payload[baseA],
        payload[baseB],
        pair.t
      );

      const py = THREE.MathUtils.lerp(
        payload[baseA + 1],
        payload[baseB + 1],
        pair.t
      );

      const pz = THREE.MathUtils.lerp(
        payload[baseA + 2],
        payload[baseB + 2],
        pair.t
      );

      mesh.position.set(px, py, pz);

      const vx = THREE.MathUtils.lerp(
        payload[baseA + 3],
        payload[baseB + 3],
        pair.t
      );

      const vy = THREE.MathUtils.lerp(
        payload[baseA + 4],
        payload[baseB + 4],
        pair.t
      );

      const vz = THREE.MathUtils.lerp(
        payload[baseA + 5],
        payload[baseB + 5],
        pair.t
      );

      const speedSq = vx * vx + vy * vy + vz * vz;

      if (speedSq > 1e-12) {
        const inverseSpeed = 1 / Math.sqrt(speedSq);

        direction.set(
          vx * inverseSpeed,
          vy * inverseSpeed,
          vz * inverseSpeed
        );

        mesh.quaternion.setFromUnitVectors(forward, direction);
      }
    }

    const bounds = calculateFlockBounds(
      this.birdMeshes,
      activeCount
    );

    return {
      activeCount,
      centroid: bounds.centroid,
      radius: bounds.radius,
    };
  }


  updateGlows(simulationTime, divergenceTime) {
    for (let i = 0; i < this.glows.length; i++) {
      const glow = this.glows[i];
      const birth = this.data.birth_events[i];

      if (!birth) {
        glow.visible = false;
        continue;
      }

      const age = simulationTime - birth.time;

      const duration = GLOW_VIEW_SECONDS * PLAYBACK_SPEED;

      if (age < 0 || age > duration) {
        glow.visible = false;
        continue;
      }

      const bird = this.birdMeshes[i];

      if (!bird.visible) {
        glow.visible = false;
        continue;
      }

      glow.visible = true;
      glow.position.copy(bird.position);

      const progress = THREE.MathUtils.clamp(
        age / duration,
        0,
        1
      );

      const pulse = 1.0 + 0.08 * Math.sin(progress * Math.PI);

      glow.scale.set(
        GLOW_SIZE * pulse,
        GLOW_SIZE * pulse,
        1
      );

      glow.material.opacity =
        0.40 * Math.pow(1 - progress, 1.15);
    }
  }


  follow(centroid, desiredDistance) {
    const oldTarget = this.controls.target.clone();

    const delta = centroid
      .clone()
      .sub(oldTarget)
      .multiplyScalar(0.08);

    this.controls.target.add(delta);
    this.camera.position.add(delta);

    const offset = this.camera.position
      .clone()
      .sub(this.controls.target);

    const currentDistance = offset.length();

    if (currentDistance > 1e-5) {
      const nextDistance = THREE.MathUtils.lerp(
        currentDistance,
        desiredDistance,
        0.025
      );

      offset.normalize().multiplyScalar(nextDistance);

      this.camera.position
        .copy(this.controls.target)
        .add(offset);
    }
  }


  bindPointer() {
    this.renderer.domElement.addEventListener(
      "pointermove",
      event => {
        this.pointerInside = true;

        const rect = this.renderer.domElement.getBoundingClientRect();

        this.pointer.x =
          ((event.clientX - rect.left) / rect.width) * 2 - 1;

        this.pointer.y =
          -((event.clientY - rect.top) / rect.height) * 2 + 1;

        this.pointerX = event.clientX;
        this.pointerY = event.clientY;
      }
    );

    this.renderer.domElement.addEventListener(
      "pointerleave",
      () => {
        this.pointerInside = false;
        this.pointer.set(100, 100);

        if (this.hovered) {
          this.hovered.scale.setScalar(1);
          this.hovered = null;
        }

        tooltip.classList.remove("visible");
      }
    );
  }


  updateHover() {
    if (!this.pointerInside) {
      return;
    }

    this.raycaster.setFromCamera(
      this.pointer,
      this.camera
    );

    const visible = this.birdMeshes.filter(
      mesh => mesh.visible
    );

    const hits = this.raycaster.intersectObjects(
      visible,
      false
    );

    if (hits.length === 0) {
      if (this.hovered) {
        this.hovered.scale.setScalar(1);
        this.hovered = null;
      }

      tooltip.classList.remove("visible");
      return;
    }

    const mesh = hits[0].object;

    if (this.hovered !== mesh) {
      if (this.hovered) {
        this.hovered.scale.setScalar(1);
      }

      this.hovered = mesh;
      mesh.scale.setScalar(1.65);
    }

    const bird = mesh.userData.bird;

    tooltipToken.textContent = readableToken(bird.token);
    tooltipIndex.textContent =
      `token ${bird.index} · id ${bird.token_id}`;

    tooltip.style.left = `${this.pointerX + 12}px`;
    tooltip.style.top = `${this.pointerY + 12}px`;

    tooltip.classList.add("visible");
  }


  updateNewestToken(activeCount) {
    const box = document.querySelector(
      this.runName === "A"
        ? "#newest-a"
        : "#newest-b"
    );

    if (!box) {
      return;
    }

    const value = box.querySelector(".newest-token-value");

    if (activeCount <= 0) {
      value.textContent = "—";
      return;
    }

    const bird = this.data.birds[activeCount - 1];

    value.textContent = readableToken(
      bird?.token ?? ""
    );
  }


  resize() {
    const width = Math.max(this.container.clientWidth, 1);
    const height = Math.max(this.container.clientHeight, 1);

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();

    this.renderer.setSize(width, height, false);
  }


  render() {
    this.controls.update();
    this.updateHover();
    this.renderer.render(this.scene, this.camera);
  }
}


// ============================================================
// LINK CAMERA ORIENTATION
// ============================================================

function linkCameraOrientations(panelA, panelB) {
  let syncing = false;

  function copyOrientation(source, destination) {
    if (syncing) {
      return;
    }

    syncing = true;

    const sourceOffset = source.camera.position
      .clone()
      .sub(source.controls.target);

    const destinationDistance = destination.camera.position
      .distanceTo(destination.controls.target);

    if (sourceOffset.lengthSq() > 1e-12) {
      const direction = sourceOffset.normalize();

      destination.camera.position
        .copy(destination.controls.target)
        .add(direction.multiplyScalar(destinationDistance));
    }

    syncing = false;
  }

  panelA.controls.addEventListener(
    "change",
    () => copyOrientation(panelA, panelB)
  );

  panelB.controls.addEventListener(
    "change",
    () => copyOrientation(panelB, panelA)
  );
}


// ============================================================
// START
// ============================================================

function getBirthTimes(run) {
  const events = run.birth_events ?? [];

  return events.map(
    event => Number(event.time ?? 0)
  );
}


function intervalForToken(
  run,
  birthTimes,
  tokenIndex
) {
  const start = birthTimes[tokenIndex] ?? 0;

  if (tokenIndex + 1 < birthTimes.length) {
    const end = birthTimes[tokenIndex + 1];

    return {
      start,
      end,
      duration: Math.max(0, end - start),
    };
  }

  const end = Number(run.duration ?? start);

  return {
    start,
    end,
    duration: Math.max(0, end - start),
  };
}


async function start() {
  const data = await loadData();

  const generationA =
    document.querySelector("#generation-a");

  const generationB =
    document.querySelector("#generation-b");


  function buildGeneration(
    container,
    birds
  ) {
    const fragment =
      document.createDocumentFragment();

    birds.forEach(
      (bird, index) => {
        const span =
          document.createElement("span");

        span.className = "generation-token";
        span.dataset.tokenIndex =
          String(index);
        span.textContent =
          bird.token ?? "";

        fragment.appendChild(span);
      }
    );

    container.replaceChildren(fragment);
  }


  buildGeneration(
    generationA,
    data.A.birds
  );

  buildGeneration(
    generationB,
    data.B.birds
  );


  let highlightedGenerationToken = -1;


  function updateGenerationHighlight(
    tokenIndex
  ) {
    if (
      tokenIndex ===
      highlightedGenerationToken
    ) {
      return;
    }

    if (
      highlightedGenerationToken >= 0
    ) {
      generationA.children[
        highlightedGenerationToken
      ]?.classList.remove(
        "current-token"
      );

      generationB.children[
        highlightedGenerationToken
      ]?.classList.remove(
        "current-token"
      );
    }

    generationA.children[
      tokenIndex
    ]?.classList.add(
      "current-token"
    );

    generationB.children[
      tokenIndex
    ]?.classList.add(
      "current-token"
    );

    highlightedGenerationToken =
      tokenIndex;
  }

  console.log("Compact trajectory loaded:", {
    A: {
      frames: data.A.frame_count,
      binaryMB: (
        data.A._buffer.byteLength / 1024 / 1024
      ).toFixed(1),
    },
    B: {
      frames: data.B.frame_count,
      binaryMB: (
        data.B._buffer.byteLength / 1024 / 1024
      ).toFixed(1),
    },
  });

  const panelA = new FlockPanel(
    document.querySelector("#scene-a"),
    "A",
    data.A
  );

  const panelB = new FlockPanel(
    document.querySelector("#scene-b"),
    "B",
    data.B
  );

  linkCameraOrientations(panelA, panelB);

  const divergenceIndex = Number(
    data.A.first_divergence
  );

  const birthAtDivergence =
    data.A.birth_events[divergenceIndex];

  if (!birthAtDivergence) {
    throw new Error(
      `No birth event exists for divergence index ${divergenceIndex}.`
    );
  }

  const divergenceTime = Number(
    birthAtDivergence.time
  );

  const birthTimesA =
    getBirthTimes(data.A);

  const birthTimesB =
    getBirthTimes(data.B);

  const commonTokenCount = Math.min(
    data.A.birds.length,
    data.B.birds.length
  );

  let playing = true;
  let sharedTokenIndex = 0;
  let intervalElapsed = 0;
  let previousRealTime = performance.now();


  playPauseButton.addEventListener(
    "click",
    () => {
      playing = !playing;

      playPauseButton.textContent =
        playing ? "Pause" : "Play";

      previousRealTime = performance.now();
    }
  );


  rewindButton.addEventListener(
    "click",
    () => {
      sharedTokenIndex = 0;
      intervalElapsed = 0;
      playing = false;
      playPauseButton.textContent = "Play";
      previousRealTime = performance.now();
      tooltip.classList.remove("visible");
    }
  );


  window.addEventListener(
    "resize",
    () => {
      panelA.resize();
      panelB.resize();
    }
  );


  function animate(now) {
    requestAnimationFrame(animate);

    const realDelta = Math.min(
      (now - previousRealTime) / 1000,
      0.1
    );

    previousRealTime = now;

    if (playing) {
      intervalElapsed +=
        realDelta * PLAYBACK_SPEED;

      // Both sides remain on the same generated-token index.
      // Carry overshoot forward so there is no one-frame stop
      // at a token boundary.
      while (
        sharedTokenIndex <
        commonTokenCount
      ) {
        const stepA =
          intervalForToken(
            data.A,
            birthTimesA,
            sharedTokenIndex
          );

        const stepB =
          intervalForToken(
            data.B,
            birthTimesB,
            sharedTokenIndex
          );

        const stepDuration = Math.max(
          stepA.duration,
          stepB.duration,
          1e-6
        );

        if (
          intervalElapsed <
          stepDuration
        ) {
          break;
        }

        if (
          sharedTokenIndex >=
          commonTokenCount - 1
        ) {
          intervalElapsed =
            stepDuration;

          playing = false;

          playPauseButton.textContent =
            "Play";

          break;
        }

        intervalElapsed -=
          stepDuration;

        sharedTokenIndex += 1;
      }
    }

    const intervalA =
      intervalForToken(
        data.A,
        birthTimesA,
        sharedTokenIndex
      );

    const intervalB =
      intervalForToken(
        data.B,
        birthTimesB,
        sharedTokenIndex
      );

    const sharedIntervalDuration =
      Math.max(
        intervalA.duration,
        intervalB.duration,
        1e-6
      );

    // Smooth sync: each run moves continuously through its
    // recorded response for this token. The shorter interval is
    // stretched only enough to meet the longer one at the next
    // shared token boundary.
    const intervalProgress =
      THREE.MathUtils.clamp(
        intervalElapsed /
          sharedIntervalDuration,
        0,
        1
      );

    const timeA =
      THREE.MathUtils.lerp(
        intervalA.start,
        intervalA.end,
        intervalProgress
      );

    const timeB =
      THREE.MathUtils.lerp(
        intervalB.start,
        intervalB.end,
        intervalProgress
      );

    const stateA = panelA.update(timeA);
    const stateB = panelB.update(timeB);

    panelA.updateNewestToken(stateA.activeCount);
    panelB.updateNewestToken(stateB.activeCount);

    panelA.updateGlows(timeA, divergenceTime);
    panelB.updateGlows(timeB, divergenceTime);

    const maxRadius = Math.max(
      stateA.radius,
      stateB.radius,
      1.0
    );

    const desiredDistance = Math.max(
      MIN_CAMERA_DISTANCE,
      maxRadius * CAMERA_RADIUS_MULTIPLIER
    );

    panelA.follow(
      stateA.centroid,
      desiredDistance
    );

    panelB.follow(
      stateB.centroid,
      desiredDistance
    );

    const currentTokenCount =
      sharedTokenIndex + 1;

    updateGenerationHighlight(
      sharedTokenIndex
    );

    if (
      sharedTokenIndex <
      divergenceIndex
    ) {
      statusDisplay.textContent =
        `shared trajectory · ` +
        `${currentTokenCount} tokens`;
    } else {
      statusDisplay.textContent =
        `diverged at token ` +
        `${divergenceIndex} · ` +
        `${currentTokenCount} tokens ` +
        `in each flock`;
    }

    timeDisplay.textContent =
      `token ${currentTokenCount} / ` +
      `${commonTokenCount}`;

    panelA.render();
    panelB.render();
  }

  requestAnimationFrame(animate);
}


start().catch(
  error => {
    console.error(error);
    statusDisplay.textContent =
      `Failed to load compact trajectory: ${error.message}`;
  }
);
