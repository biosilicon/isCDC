const TYPES = {bool: Uint8Array, int8: Int8Array, uint8: Uint8Array, int16: Int16Array,
  uint16: Uint16Array, int32: Int32Array, uint32: Uint32Array, int64: BigInt64Array,
  uint64: BigUint64Array, float32: Float32Array, float64: Float64Array};
export const VIRIDIS = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
export const DIVERGING = [[33, 102, 172], [247, 247, 247], [178, 24, 43]];

export function decodeMolecularArray(buffer, expectedKind) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 32) throw new Error("Short molecular array");
  const bytes = new Uint8Array(buffer);
  const magic = [73, 83, 67, 68, 67, 77, 79, 0];
  if (magic.some((value, i) => bytes[i] !== value)) throw new Error("Invalid molecular magic");
  const header = new DataView(buffer);
  const version = header.getUint16(8, true), kind = header.getUint16(10, true);
  const count = header.getUint32(12, true);
  const dtype = new TextDecoder().decode(bytes.subarray(16, 32)).replace(/\0+$/, "");
  const Type = TYPES[dtype];
  if (version !== 1 || kind !== expectedKind || !Type || ![1, 2].includes(kind)) {
    throw new Error("Unsupported molecular array");
  }
  const elements = kind === 1 ? 2 * count : count;
  const dataBytes = elements * Type.BYTES_PER_ELEMENT;
  if (buffer.byteLength !== 32 + dataBytes + (kind === 2 ? count : 0)) {
    throw new Error("Molecular array size mismatch");
  }
  if (kind === 1) {
    if (dtype !== "float64") throw new Error("Invalid coordinate type");
    const x = new Float64Array(buffer, 32, count);
    const y = new Float64Array(buffer, 32 + count * 8, count);
    for (let i = 0; i < count; i++) {
      if (!Number.isFinite(x[i]) || !Number.isFinite(y[i])) throw new Error("Non-finite coordinate");
    }
    return {kind: "molecular", count, x, y};
  }
  const values = new Type(buffer, 32, count);
  const states = new Uint8Array(buffer, 32 + dataBytes, count);
  for (let i = 0; i < count; i++) {
    if (states[i] > 2 || (states[i] === 0 && !Number.isFinite(Number(values[i])))) {
      throw new Error("Invalid molecular measurement state");
    }
    if (dtype === "bool" && values[i] > 1) throw new Error("Invalid boolean value");
  }
  return {count, values, states, dtype};
}

export function molecularRange(points, valueType, requestedScale = "log1p") {
  let min = Infinity, max = -Infinity, valid = 0, missing = 0, invalid = 0;
  for (let i = 0; i < points.count; i++) {
    if (points.states[i] === 1) { missing++; continue; }
    if (points.states[i] === 2) { invalid++; continue; }
    const value = Number(points.values[i]);
    min = Math.min(min, value); max = Math.max(max, value); valid++;
  }
  const constant = valid > 0 && min === max;
  const logAllowed = valid > 0 && min >= 0 && !["binary", "log_normalized"].includes(valueType);
  const scale = requestedScale === "log1p" && logAllowed ? "log1p" : "linear";
  const diverging = min < 0 && max > 0;
  if (!valid) { min = 0; max = 0; }
  if (valueType === "binary") { min = 0; max = 1; }
  if (diverging) { max = Math.max(-min, max); min = -max; }
  return {min, max, valid, missing, invalid, constant, logAllowed, scale, diverging};
}

export function interpolateColor(t, palette) {
  const location = Math.max(0, Math.min(1, t)) * (palette.length - 1);
  const left = Math.min(Math.floor(location), palette.length - 2), fraction = location - left;
  return [...palette[left].map((value, i) => Math.round(value + (palette[left + 1][i] - value) * fraction)), 230];
}

export function molecularAttributes(points, range, yAxis = "up") {
  const positions = new Float32Array(points.count * 2);
  const colors = new Uint8Array(points.count * 4);
  const radii = new Float32Array(points.count).fill(2.25);
  const transform = range.scale === "log1p" ? Math.log1p : (value) => value;
  const low = transform(range.min), span = transform(range.max) - low;
  const palette = range.diverging ? DIVERGING : VIRIDIS;
  // Draw missing observations first, then low-to-high signal, so measured zeros
  // cannot hide a rare high-value point. Preserve the source index for picking.
  if (!points.renderOrder) {
    const order = Uint32Array.from({length: points.count}, (_, i) => i);
    const weights = Float64Array.from(points.values, (value, i) => points.states[i] !== 0
      ? -Infinity : range.diverging ? Math.abs(Number(value)) : Number(value));
    order.sort((a, b) => {
      if (weights[a] === weights[b]) return a - b;
      return weights[a] < weights[b] ? -1 : 1;
    });
    points.renderOrder = order;
  }
  const sourceIndices = points.renderOrder;
  const lut = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) lut.set(interpolateColor(i / 255, palette), i * 4);
  for (let i = 0; i < points.count; i++) {
    const j = sourceIndices[i];
    positions[i * 2] = points.x[j];
    positions[i * 2 + 1] = yAxis === "down" ? -points.y[j] : points.y[j];
    if (points.states[j] !== 0) {
      colors.set(points.states[j] === 1 ? [148, 163, 184, 100] : [196, 177, 177, 160], i * 4);
    } else {
      const t = span === 0 ? (range.min === 0 ? 0 : 0.5)
        : (transform(Number(points.values[j])) - low) / span;
      const color = Math.round(Math.max(0, Math.min(1, t)) * 255) * 4;
      for (let channel = 0; channel < 4; channel++) colors[i * 4 + channel] = lut[color + channel];
    }
  }
  return {positions, colors, radii, sourceIndices};
}

export function molecularHover(points, index) {
  if (!points || index < 0 || index >= points.count) return "";
  const value = points.states[index] === 1 ? "Not measured in this modality"
    : points.states[index] === 2 ? "Non-finite stored value" : String(points.values[index]);
  return `${points.feature.label} · ${points.feature.id}\n${points.modality.name} · ${points.modality.value_type}\n${points.sampleId} · (${points.x[index]}, ${points.y[index]})\n${value}`;
}
