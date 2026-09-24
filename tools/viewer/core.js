// Pure data-decoding and face-categorization logic for the results viewer.
// No DOM, no three.js -- this module is exercised directly (via the
// window.__viewerCore handle viewer.js attaches) by scripts/check_viewer.py
// so the exact shipped categorization code is what gets checked against
// the Python-computed reference counts, not a reimplementation of it.
//
// Category ids (single-config view):
//   0 correctly_flagged_unobserved  (gt unobserved, predicted unobserved)
//   1 false_reassurance             (gt unobserved, predicted observed)
//   2 false_alarm                   (gt observed, predicted unobserved, not ignore)
//   3 correctly_observed            (gt observed, predicted observed, not ignore)
//   4 ignore_set                    (ignore set, regardless of prediction)
//
// Diff-mode category ids (two configs, same tau, raw predicted-unobserved,
// no ignore-set handling -- matches Stage 1's check C definition exactly):
//   0 only_a   1 only_b   2 both   3 neither

export const CATEGORY_NAMES = [
  "correctly_flagged_unobserved",
  "false_reassurance",
  "false_alarm",
  "correctly_observed",
  "ignore_set",
];

export const DIFF_CATEGORY_NAMES = ["only_a", "only_b", "both", "neither"];

export function base64ToBytes(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

export async function gunzip(bytes) {
  const ds = new DecompressionStream("gzip");
  const stream = new Blob([bytes]).stream().pipeThrough(ds);
  const buf = await new Response(stream).arrayBuffer();
  return new Uint8Array(buf);
}

export async function decodeGzipBase64(b64) {
  return gunzip(base64ToBytes(b64));
}

const DTYPE_CTORS = {
  float32: Float32Array,
  float64: Float64Array,
  int32: Int32Array,
  uint8: Uint8Array,
};

// bytes: Uint8Array of raw little-endian data. Assumes a little-endian
// host (true of every browser JS engine in practice) -- no byte-swap.
export function bytesToTypedArray(bytes, dtype) {
  const Ctor = DTYPE_CTORS[dtype];
  if (!Ctor) throw new Error(`unknown dtype ${dtype}`);
  if (bytes.byteOffset % Ctor.BYTES_PER_ELEMENT !== 0) {
    bytes = bytes.slice();
  }
  return new Ctor(bytes.buffer, bytes.byteOffset, bytes.byteLength / Ctor.BYTES_PER_ELEMENT);
}

// Unpack bits from a Uint8Array with numpy's packbits(bitorder="little")
// convention: within byte b, element (8*byteIdx + bitIdx) is (b >> bitIdx) & 1.
export function unpackBitsLittle(bytes, nBits) {
  const out = new Uint8Array(nBits);
  for (let i = 0; i < nBits; i++) {
    const byteIdx = i >> 3;
    const bitIdx = i & 7;
    out[i] = (bytes[byteIdx] >> bitIdx) & 1;
  }
  return out;
}

// packed: Uint8Array flattened from (nConfigs, nTaus, nBytesPerRow), row-major.
// Returns a Uint8Array (0/1) of length nFaces: predicted_observed for
// (configIdx, tauIdx).
export function getPredictedObserved(packed, configIdx, tauIdx, nTaus, nBytesPerRow, nFaces) {
  const rowStart = (configIdx * nTaus + tauIdx) * nBytesPerRow;
  const row = packed.subarray(rowStart, rowStart + nBytesPerRow);
  return unpackBitsLittle(row, nFaces);
}

// gtObserved, ignoreSet, predictedObserved: Uint8Array (0/1), length nFaces.
// Returns Uint8Array of category ids (see module docstring).
export function categorizeFaces(nFaces, gtObserved, ignoreSet, predictedObserved) {
  const out = new Uint8Array(nFaces);
  for (let i = 0; i < nFaces; i++) {
    if (ignoreSet[i]) {
      out[i] = 4;
      continue;
    }
    const gtObs = gtObserved[i] !== 0;
    const predObs = predictedObserved[i] !== 0;
    if (!gtObs && !predObs) out[i] = 0; // correctly flagged unobserved
    else if (!gtObs && predObs) out[i] = 1; // false reassurance
    else if (gtObs && !predObs) out[i] = 2; // false alarm
    else out[i] = 3; // correctly observed
  }
  return out;
}

// predictedObservedA/B: Uint8Array (0/1), length nFaces, same tau, two configs.
export function diffCategorizeFaces(nFaces, predictedObservedA, predictedObservedB) {
  const out = new Uint8Array(nFaces);
  for (let i = 0; i < nFaces; i++) {
    const puA = predictedObservedA[i] === 0; // predicted UNobserved
    const puB = predictedObservedB[i] === 0;
    if (puA && !puB) out[i] = 0;
    else if (puB && !puA) out[i] = 1;
    else if (puA && puB) out[i] = 2;
    else out[i] = 3;
  }
  return out;
}

export function countCategories(categories, nCategories) {
  const counts = new Array(nCategories).fill(0);
  for (let i = 0; i < categories.length; i++) counts[categories[i]]++;
  return counts;
}

// metrics.json (written by scripts/export_viewer_data.py) contains bare
// `NaN` tokens (Python's json.dump default) for the empty "medium" size
// class's recall, which is not valid JSON. Replace the bare token with a
// quoted sentinel before parsing, then restore it as JS NaN via reviver.
export function parseJsonWithNaN(text) {
  const sentinel = "__NaN_SENTINEL__";
  const patched = text.replace(/\bNaN\b/g, `"${sentinel}"`);
  return JSON.parse(patched, (_key, value) => (value === sentinel ? NaN : value));
}
