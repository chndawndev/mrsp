// Display-only 3D viewer for the Stage 1 export of c1_cecum_t1_v1. This
// file never computes a metric: every number shown in the panel is read
// straight out of metrics.json / regions.json. The only thing computed
// here is the per-face color category, from the exported boolean arrays
// (categorizeFaces / diffCategorizeFaces, both in core.js above).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// Okabe-Ito colorblind-safe palette.
const CATEGORY_COLORS = [
  [0x00 / 255, 0x72 / 255, 0xb2 / 255], // 0 correctly flagged unobserved -- blue
  [0xd5 / 255, 0x5e / 255, 0x00 / 255], // 1 false reassurance -- vermillion
  [0xe6 / 255, 0x9f / 255, 0x00 / 255], // 2 false alarm -- orange
  [0x9a / 255, 0x9a / 255, 0x9a / 255], // 3 correctly observed -- neutral gray
  [0xd9 / 255, 0xd9 / 255, 0xd9 / 255], // 4 ignore set -- light gray + hatching
];
const CATEGORY_LABELS = [
  "Correctly flagged unobserved",
  "False reassurance (GT unobserved, predicted observed)",
  "False alarm (GT observed, predicted unobserved)",
  "Correctly observed",
  "Ignore set (structurally unreachable)",
];
const DIFF_COLORS = [
  [0xe6 / 255, 0x9f / 255, 0x00 / 255], // only A -- orange
  [0x56 / 255, 0xb4 / 255, 0xe9 / 255], // only B -- sky blue
  [0xcc / 255, 0x79 / 255, 0xa7 / 255], // both -- reddish purple
  [0xd9 / 255, 0xd9 / 255, 0xd9 / 255], // neither -- light gray
];
const DIFF_LABELS = ["Only A", "Only B", "Both", "Neither"];
const HEADLINE_DIM_COLOR = [0xd9 / 255, 0xd9 / 255, 0xd9 / 255];

function readDataBlock(id) {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing data block ${id}`);
  return el.textContent.trim();
}

async function loadTypedArray(key, manifest) {
  const info = manifest.files[key];
  const bytes = await decodeGzipBase64(readDataBlock(`data-${key}`));
  return bytesToTypedArray(bytes, info.dtype);
}

async function loadJson(key) {
  const bytes = await decodeGzipBase64(readDataBlock(`data-${key}`));
  const text = new TextDecoder("utf-8").decode(bytes);
  return parseJsonWithNaN(text);
}

async function boot() {
  const manifest = JSON.parse(document.getElementById("manifest-data").textContent);

  const [verticesF32, facesI32, gtObservedU8, ignoreSetU8, gtRegionIdI32, packed] = await Promise.all([
    loadTypedArray("vertices_f32", manifest),
    loadTypedArray("faces_i32", manifest),
    loadTypedArray("gt_observed_u8", manifest),
    loadTypedArray("ignore_set_u8", manifest),
    loadTypedArray("gt_region_id_i32", manifest),
    loadTypedArray("predicted_observed_packed", manifest),
  ]);
  const [metrics, regions, trajectory] = await Promise.all([
    loadJson("metrics_json"),
    loadJson("regions_json"),
    loadJson("trajectory_json"),
  ]);

  const nFaces = manifest.n_faces;
  const nVerts = manifest.n_vertices;
  const configNames = manifest.configurations;
  const taus = manifest.taus;
  const nBytesPerRow = manifest.files.predicted_observed_packed.shape[2];

  const headlineIds = new Set(regions.filter((r) => r.headline).map((r) => r.id));
  const headlineFaceMask = new Uint8Array(nFaces);
  for (let i = 0; i < nFaces; i++) {
    const rid = gtRegionIdI32[i];
    if (rid >= 0 && headlineIds.has(rid)) headlineFaceMask[i] = 1;
  }
  const classIndexById = new Map();
  {
    const byClass = { small: [], medium: [], large: [], below_headline: [] };
    for (const r of regions) byClass[r.size_class].push(r.id);
    for (const cls of Object.keys(byClass)) {
      byClass[cls].forEach((id, idx) => classIndexById.set(id, idx));
    }
  }
  const regionById = new Map(regions.map((r) => [r.id, r]));

  function predictedObserved(configName, tau) {
    const ci = configNames.indexOf(configName);
    const ti = taus.indexOf(tau);
    return getPredictedObserved(packed, ci, ti, taus.length, nBytesPerRow, nFaces);
  }

  // ---------------- three.js scene ----------------
  const wrap = document.getElementById("canvas-wrap");
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(wrap.clientWidth, wrap.clientHeight);
  wrap.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x14171a);

  const camera = new THREE.PerspectiveCamera(50, wrap.clientWidth / wrap.clientHeight, 0.1, 10000);
  camera.position.set(0, 0, 300);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  // Non-indexed BufferGeometry: one expanded vertex triple per face, so
  // faceIndex from the raycaster equals our face id directly, and colors
  // can be set independently per face (a shared-vertex indexed geometry
  // could not do that without splitting vertices anyway).
  const positions = new Float32Array(nFaces * 3 * 3);
  for (let f = 0; f < nFaces; f++) {
    for (let v = 0; v < 3; v++) {
      const vi = facesI32[f * 3 + v];
      positions[(f * 3 + v) * 3 + 0] = verticesF32[vi * 3 + 0];
      positions[(f * 3 + v) * 3 + 1] = verticesF32[vi * 3 + 1];
      positions[(f * 3 + v) * 3 + 2] = verticesF32[vi * 3 + 2];
    }
  }
  const colors = new Float32Array(nFaces * 3 * 3);
  const hatch = new Float32Array(nFaces * 3);
  for (let f = 0; f < nFaces; f++) {
    const h = ignoreSetU8[f] ? 1 : 0;
    hatch[f * 3] = h;
    hatch[f * 3 + 1] = h;
    hatch[f * 3 + 2] = h;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute("hatch", new THREE.BufferAttribute(hatch, 1));
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();

  const material = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
  material.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nattribute float hatch;\nvarying float vHatch;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvHatch = hatch;");
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", "#include <common>\nvarying float vHatch;")
      .replace(
        "#include <color_fragment>",
        "#include <color_fragment>\nif (vHatch > 0.5) { float stripe = mod(gl_FragCoord.x + gl_FragCoord.y, 8.0); if (stripe < 4.0) diffuseColor.rgb *= 0.6; }"
      );
  };
  const mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);

  if (geometry.boundingSphere) {
    // A diagonal offset (not a single axis) clears an elongated tubular
    // mesh whose long axis happens to align with one world axis -- a
    // pure +Z offset can leave the camera inside the lumen.
    const c = geometry.boundingSphere.center;
    const r = geometry.boundingSphere.radius;
    camera.position.set(c.x + r * 2.8, c.y + r * 2.0, c.z + r * 2.8);
    camera.far = r * 20;
    camera.updateProjectionMatrix();
    controls.target.copy(c);
    controls.update();
  }

  // Trajectory polylines.
  function trajLine(key, color) {
    const pts = trajectory.frames.map((f) => new THREE.Vector3(...f[key]));
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color }));
    scene.add(line);
    return line;
  }
  trajLine("gt_position_mm", 0xffffff);
  trajLine("pred_position_mm", 0x4da3ff);

  const gtMarker = new THREE.Mesh(new THREE.SphereGeometry(1.5, 12, 12), new THREE.MeshBasicMaterial({ color: 0xffffff }));
  const predMarker = new THREE.Mesh(new THREE.SphereGeometry(1.5, 12, 12), new THREE.MeshBasicMaterial({ color: 0x4da3ff }));
  scene.add(gtMarker, predMarker);
  let gtArrow = null;
  let predArrow = null;

  let regionLine = null;

  // ---------------- coloring ----------------
  let state = {
    config: configNames[0],
    tau: taus.includes(0.25) ? 0.25 : taus[0],
    headlineOnly: false,
    diffMode: false,
    diffA: configNames[0],
    diffB: configNames.length > 1 ? configNames[1] : configNames[0],
  };

  function applyColor(faceIdx, rgb) {
    for (let v = 0; v < 3; v++) {
      const base = (faceIdx * 3 + v) * 3;
      colors[base] = rgb[0];
      colors[base + 1] = rgb[1];
      colors[base + 2] = rgb[2];
    }
  }

  function recolor() {
    let categories;
    let palette, labels;
    if (state.diffMode) {
      const puA = predictedObserved(state.diffA, state.tau);
      const puB = predictedObserved(state.diffB, state.tau);
      categories = diffCategorizeFaces(nFaces, puA, puB);
      palette = DIFF_COLORS;
      labels = DIFF_LABELS;
    } else {
      const po = predictedObserved(state.config, state.tau);
      categories = categorizeFaces(nFaces, gtObservedU8, ignoreSetU8, po);
      palette = CATEGORY_COLORS;
      labels = CATEGORY_LABELS;
    }
    for (let f = 0; f < nFaces; f++) {
      if (state.headlineOnly && !headlineFaceMask[f]) {
        applyColor(f, HEADLINE_DIM_COLOR);
      } else {
        applyColor(f, palette[categories[f]]);
      }
    }
    geometry.attributes.color.needsUpdate = true;

    const counts = countCategories(categories, palette.length);
    renderLegend(labels, palette, counts);
    return { categories, counts, labels };
  }

  // ---------------- legend ----------------
  function rgbToCss(rgb) {
    return `rgb(${Math.round(rgb[0] * 255)},${Math.round(rgb[1] * 255)},${Math.round(rgb[2] * 255)})`;
  }
  function renderLegend(labels, palette, counts) {
    const el = document.getElementById("legend");
    el.innerHTML = "";
    labels.forEach((label, i) => {
      const row = document.createElement("div");
      row.className = "legend-item";
      row.innerHTML = `<span class="swatch" style="background:${rgbToCss(palette[i])}"></span><span>${label}: ${counts[i].toLocaleString()}</span>`;
      el.appendChild(row);
    });
  }

  // ---------------- metrics panel ----------------
  function fmtNum(v) {
    if (v === null || v === undefined) return "n/a";
    if (typeof v === "number") {
      if (Number.isNaN(v)) return "n/a";
      if (!Number.isInteger(v)) return v.toFixed(6);
      return v.toLocaleString();
    }
    return String(v);
  }

  function renderMetricsPanel() {
    const el = document.getElementById("metrics-panel");
    const cfg = metrics.configurations[state.config];
    const tauOut = cfg.by_tau[String(state.tau)];
    let html = "";
    html += `<table class="kv">`;
    html += `<tr><td>independent meshes</td><td>1</td></tr>`;
    html += `<tr><td>independent trajectories</td><td>1</td></tr>`;
    html += `<tr><td colspan="2" class="note">upper bound: GT scale and Sim(3) alignment used</td></tr>`;
    html += `<tr><td>depth_scale_median</td><td>${fmtNum(metrics.depth_scale_median)}</td></tr>`;
    html += `<tr><td>pose_alignment_s_pose</td><td>${fmtNum(metrics.pose_alignment_s_pose)}</td></tr>`;
    html += `<tr><td>ray_miss_frac</td><td>${fmtNum(cfg.ray_miss_frac)}</td></tr>`;
    html += `<tr><td>evaluable_frac</td><td>${fmtNum(cfg.evaluable_frac)}</td></tr>`;
    html += `<tr><td>d_pred_unavailable_count</td><td>${fmtNum(cfg.d_pred_unavailable_count)}</td></tr>`;
    for (const [k, v] of Object.entries(tauOut)) {
      if (k === "detection_sweep" || k === "recall_at_50pct") continue;
      html += `<tr><td>${k}</td><td>${fmtNum(v)}</td></tr>`;
    }
    html += `</table>`;

    html += `<div class="note">Region recall @ 50%</div>`;
    html += `<table class="sweep"><tr><th>class</th><th>n</th><th>detected</th><th>recall</th></tr>`;
    for (const [cls, v] of Object.entries(tauOut.recall_at_50pct)) {
      html += `<tr><td>${cls}</td><td>${v.n_regions}</td><td>${v.n_detected}</td><td>${fmtNum(v.recall)}</td></tr>`;
    }
    html += `</table>`;

    html += `<div class="note">Detection sweep (25/50/75%)</div>`;
    html += `<table class="sweep"><tr><th>thresh</th><th>class</th><th>recall</th></tr>`;
    for (const [thresh, byClass] of Object.entries(tauOut.detection_sweep)) {
      for (const [cls, v] of Object.entries(byClass)) {
        html += `<tr><td>${thresh}</td><td>${cls}</td><td>${fmtNum(v.recall)}</td></tr>`;
      }
    }
    html += `</table>`;

    html += `<div class="note">Headline GT regions</div>`;
    html += `<table class="sweep"><tr><th>region</th><th>coverage</th><th>detected@50</th><th>loc.err (mm)</th></tr>`;
    for (const r of regions.filter((x) => x.headline).sort((a, b) => a.id - b.id)) {
      const t = r.by_config_by_tau[state.config][String(state.tau)];
      const label = `id ${r.id} (${r.size_class} #${classIndexById.get(r.id)})`;
      html += `<tr><td>${label}</td><td>${fmtNum(t.coverage_fraction)}</td><td>${t.detected_at["0.5"]}</td><td>${fmtNum(t.localization_error_mm)}</td></tr>`;
    }
    html += `</table>`;

    el.innerHTML = html;
  }

  // ---------------- region click readout ----------------
  function showRegionInfo(faceIdx, category) {
    const el = document.getElementById("region-info");
    const rid = gtRegionIdI32[faceIdx];
    let html = `<table class="kv">`;
    html += `<tr><td>face id</td><td>${faceIdx}</td></tr>`;
    html += `<tr><td>category</td><td>${state.diffMode ? DIFF_LABELS[category] : CATEGORY_LABELS[category]}</td></tr>`;
    if (rid < 0) {
      html += `<tr><td colspan="2" class="note">Not part of a GT-unobserved region.</td></tr></table>`;
      el.innerHTML = html;
      if (regionLine) { scene.remove(regionLine); regionLine = null; }
      return;
    }
    const r = regionById.get(rid);
    const label = `id ${r.id} (${r.size_class}${r.headline ? " #" + classIndexById.get(r.id) : ""})`;
    html += `<tr><td>region</td><td>${label}</td></tr>`;
    html += `<tr><td>n_faces</td><td>${r.n_faces}</td></tr>`;
    html += `<tr><td>area_mm2</td><td>${fmtNum(r.area_mm2)}</td></tr>`;
    html += `<tr><td>diameter_mm</td><td>${fmtNum(r.diameter_mm)}</td></tr>`;
    if (!r.headline) {
      html += `<tr><td colspan="2" class="note">Below-headline region: per-config coverage/localization was not exported (Stage 1 only exports these for headline, d&ge;5mm regions).</td></tr></table>`;
      el.innerHTML = html;
      if (regionLine) { scene.remove(regionLine); regionLine = null; }
      return;
    }
    const t = r.by_config_by_tau[state.config][String(state.tau)];
    html += `<tr><td>coverage_fraction</td><td>${fmtNum(t.coverage_fraction)}</td></tr>`;
    html += `<tr><td>detected@25/50/75</td><td>${t.detected_at["0.25"]} / ${t.detected_at["0.5"]} / ${t.detected_at["0.75"]}</td></tr>`;
    html += `<tr><td>localization_error_mm</td><td>${fmtNum(t.localization_error_mm)}</td></tr>`;
    html += `<tr><td>segment_intersects_mesh</td><td>${t.segment_intersects_mesh}</td></tr>`;
    html += `</table>`;
    el.innerHTML = html;

    if (regionLine) scene.remove(regionLine);
    if (t.matched_predicted_component_centroid_mm) {
      const pts = [new THREE.Vector3(...r.centroid_mm), new THREE.Vector3(...t.matched_predicted_component_centroid_mm)];
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      regionLine = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0xffe066 }));
      scene.add(regionLine);
    } else {
      regionLine = null;
    }
  }

  // ---------------- picking ----------------
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let lastCategories = null;

  renderer.domElement.addEventListener("click", (ev) => {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObject(mesh);
    if (hits.length === 0) return;
    const faceIdx = hits[0].faceIndex;
    const category = lastCategories ? lastCategories[faceIdx] : 0;
    showRegionInfo(faceIdx, category);
  });

  // ---------------- controls wiring ----------------
  const selConfig = document.getElementById("sel-config");
  const selTau = document.getElementById("sel-tau");
  const selA = document.getElementById("sel-config-a");
  const selB = document.getElementById("sel-config-b");
  configNames.forEach((c) => {
    for (const sel of [selConfig, selA, selB]) {
      const opt = document.createElement("option");
      opt.value = c;
      opt.textContent = c;
      sel.appendChild(opt);
    }
  });
  selB.value = state.diffB;
  taus.forEach((t) => {
    const opt = document.createElement("option");
    opt.value = String(t);
    opt.textContent = String(t);
    selTau.appendChild(opt);
  });
  selConfig.value = state.config;
  selTau.value = String(state.tau);

  function refresh() {
    const result = recolor();
    lastCategories = result.categories;
    renderMetricsPanel();
    return result;
  }

  selConfig.addEventListener("change", () => { state.config = selConfig.value; refresh(); });
  selTau.addEventListener("change", () => { state.tau = Number(selTau.value); refresh(); });
  selA.addEventListener("change", () => { state.diffA = selA.value; if (state.diffMode) refresh(); });
  selB.addEventListener("change", () => { state.diffB = selB.value; if (state.diffMode) refresh(); });

  document.getElementById("btn-headline").addEventListener("click", (ev) => {
    state.headlineOnly = !state.headlineOnly;
    ev.target.classList.toggle("active", state.headlineOnly);
    refresh();
  });
  document.getElementById("btn-diff").addEventListener("click", (ev) => {
    state.diffMode = !state.diffMode;
    ev.target.classList.toggle("active", state.diffMode);
    document.getElementById("diff-controls").style.display = state.diffMode ? "flex" : "none";
    refresh();
  });

  // ---------------- frame slider ----------------
  const slider = document.getElementById("frame-slider");
  const frameLabel = document.getElementById("frame-label");
  slider.max = String(trajectory.frames.length - 1);
  function updateFrame() {
    const i = Number(slider.value);
    const f = trajectory.frames[i];
    frameLabel.textContent = `${i} / ${trajectory.frames.length - 1}`;
    gtMarker.position.set(...f.gt_position_mm);
    predMarker.position.set(...f.pred_position_mm);
    if (gtArrow) scene.remove(gtArrow);
    if (predArrow) scene.remove(predArrow);
    gtArrow = new THREE.ArrowHelper(new THREE.Vector3(...f.gt_direction).normalize(), gtMarker.position, 15, 0xffffff);
    predArrow = new THREE.ArrowHelper(new THREE.Vector3(...f.pred_direction).normalize(), predMarker.position, 15, 0x4da3ff);
    scene.add(gtArrow, predArrow);
    document.getElementById("trajectory-info").textContent =
      `frame ${i}: GT pos (${f.gt_position_mm.map((x) => x.toFixed(1)).join(", ")}) -- pred pos (${f.pred_position_mm.map((x) => x.toFixed(1)).join(", ")})`;
  }
  slider.addEventListener("input", updateFrame);

  // ---------------- render loop ----------------
  function onResize() {
    camera.aspect = wrap.clientWidth / wrap.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(wrap.clientWidth, wrap.clientHeight);
  }
  window.addEventListener("resize", onResize);

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }

  document.getElementById("loading").remove();
  const initial = refresh();
  updateFrame();
  animate();

  // ---------------- check hooks (scripts/check_viewer.py) ----------------
  window.__viewerCore = { categorizeFaces, diffCategorizeFaces, countCategories, CATEGORY_NAMES, DIFF_CATEGORY_NAMES };
  window.__viewerComputeCounts = function (configName, tau) {
    const po = predictedObserved(configName, tau);
    const cats = categorizeFaces(nFaces, gtObservedU8, ignoreSetU8, po);
    return countCategories(cats, CATEGORY_COLORS.length);
  };
  window.__viewerSetState = function (partial) {
    Object.assign(state, partial);
    refresh();
  };
  console.log("[viewer-check] ready nFaces=" + nFaces);
  console.log("[viewer-check] counts oracle tau=0.25 " + JSON.stringify(window.__viewerComputeCounts("oracle", 0.25)));
}

boot().catch((err) => {
  console.error(err);
  const el = document.getElementById("loading");
  if (el) el.textContent = "Error: " + err.message;
});
