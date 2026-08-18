const FALLBACK_COLOR = [108, 117, 125, 230];

export function createRequestGate() {
  let sequence = 0;
  return {
    begin() {
      sequence += 1;
      return sequence;
    },
    isCurrent(token) {
      return token === sequence;
    },
    invalidate() {
      sequence += 1;
    },
  };
}

export function normalizeColor(value) {
  if (Array.isArray(value) && (value.length === 3 || value.length === 4)) {
    const color = value.map((part) => Math.max(0, Math.min(255, Number(part) || 0)));
    if (color.length === 3) color.push(230);
    return color;
  }
  const match = typeof value === "string" && value.match(/^#([0-9a-f]{6})([0-9a-f]{2})?$/i);
  if (!match) return [...FALLBACK_COLOR];
  const rgb = match[1];
  return [
    Number.parseInt(rgb.slice(0, 2), 16),
    Number.parseInt(rgb.slice(2, 4), 16),
    Number.parseInt(rgb.slice(4, 6), 16),
    match[2] ? Number.parseInt(match[2], 16) : 230,
  ];
}

export function categoryMap(categories) {
  return new Map(
    categories.map((category) => [Number(category.code), {
      ...category,
      code: Number(category.code),
      color: normalizeColor(category.color),
    }]),
  );
}

export function validateCategoryCodes(types, categories) {
  const allowed = new Set(categories.map((category) => Number(category.code)));
  for (let index = 0; index < types.length; index += 1) {
    if (!allowed.has(types[index])) {
      throw new Error(`Unknown category code ${types[index]} at point ${index}`);
    }
  }
}

export function buildLegendEntries(types, categories) {
  const byCode = categoryMap(categories);
  const counts = new Map();
  for (const code of types) counts.set(code, (counts.get(code) || 0) + 1);
  for (const code of counts.keys()) {
    if (!byCode.has(code)) {
      byCode.set(code, {
        code,
        label: `Unknown category ${code}`,
        color: [...FALLBACK_COLOR],
        count: 0,
        cellOntologyId: null,
      });
    }
  }
  const total = types.length;
  return [...byCode.values()].map((category) => {
    const count = counts.get(category.code) || 0;
    return {...category, count, proportion: total === 0 ? 0 : count / total};
  });
}

export function buildBinaryAttributes(points, categories, selectedCodes, yAxis = "up") {
  const byCode = categoryMap(categories);
  const positions = new Float32Array(points.count * 2);
  const colors = new Uint8Array(points.count * 4);
  const radii = new Float32Array(points.count);
  const flipY = yAxis === "down";
  for (let index = 0; index < points.count; index += 1) {
    const code = points.type[index];
    const color = byCode.get(code)?.color || FALLBACK_COLOR;
    positions[index * 2] = points.x[index];
    positions[index * 2 + 1] = flipY ? -points.y[index] : points.y[index];
    colors.set(color, index * 4);
    radii[index] = selectedCodes.has(code) ? 2.25 : 0;
  }
  return {positions, colors, radii};
}

export function coordinateBounds(positions) {
  if (positions.length === 0) return {minX: -1, maxX: 1, minY: -1, maxY: 1};
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let index = 0; index < positions.length; index += 2) {
    const x = positions[index];
    const y = positions[index + 1];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  if (!Number.isFinite(minX)) return {minX: -1, maxX: 1, minY: -1, maxY: 1};
  return {minX, maxX, minY, maxY};
}

export function resetViewState(bounds, width, height) {
  const spanX = Math.max(bounds.maxX - bounds.minX, Number.EPSILON);
  const spanY = Math.max(bounds.maxY - bounds.minY, Number.EPSILON);
  const safeWidth = Math.max(1, width);
  const safeHeight = Math.max(1, height);
  const scale = Math.max(Number.EPSILON, Math.min(safeWidth / spanX, safeHeight / spanY) * 0.9);
  return {
    target: [(bounds.minX + bounds.maxX) / 2, (bounds.minY + bounds.maxY) / 2, 0],
    zoom: Math.log2(scale),
    minZoom: -20,
    maxZoom: 24,
  };
}

export function formatHoverText(points, index, categories, annotationKind) {
  if (!points || index < 0 || index >= points.count) return "";
  const code = points.type[index];
  const label = categoryMap(categories).get(code)?.label || `Unknown category ${code}`;
  if (annotationKind !== "inferred" || points.confidence === null) return label;
  const confidence = points.confidence[index];
  return Number.isFinite(confidence) ? `${label} · confidence ${confidence.toFixed(3)}` : label;
}
