import assert from "node:assert/strict";
import test from "node:test";

import {
  buildBinaryAttributes,
  buildLegendEntries,
  coordinateBounds,
  createRequestGate,
  formatHoverText,
  initialSelectedCategoryCodes,
  resetViewState,
  validateCategoryCodes,
} from "../src/visualization_state.js";

const categories = [
  {code: 1, label: "T cell", color: "#112233"},
  {code: 2, label: "B cell", color: [4, 5, 6]},
];
const points = {
  count: 3,
  x: new Float32Array([0, 10, 20]),
  y: new Float32Array([3, 5, 7]),
  confidence: new Float32Array([0.9, 0.8, Number.NaN]),
  type: new Uint16Array([1, 2, 9]),
};

test("request gate rejects stale sample results and invalidation", () => {
  const gate = createRequestGate();
  const first = gate.begin();
  const second = gate.begin();
  assert.equal(gate.isCurrent(first), false);
  assert.equal(gate.isCurrent(second), true);
  gate.invalidate();
  assert.equal(gate.isCurrent(second), false);
});

test("legend counts known and fallback categories per sample", () => {
  const entries = buildLegendEntries(points.type, categories);
  assert.deepEqual(entries.map(({code, count}) => [code, count]), [[1, 1], [2, 1], [9, 1]]);
  assert.equal(entries[2].label, "Unknown category 9");
  assert.equal(entries[0].proportion, 1 / 3);
});

test("strict category validation rejects codes missing from configuration", () => {
  assert.doesNotThrow(() => validateCategoryCodes(new Uint16Array([1, 2]), categories));
  assert.throws(() => validateCategoryCodes(points.type, categories), /Unknown category code 9/);
});

test("Unannotated is the only category hidden by default", () => {
  const selected = initialSelectedCategoryCodes([
    ...categories,
    {code: 3, label: "Unannotated", color: "#778899"},
  ]);

  assert.deepEqual([...selected], [1, 2]);
  assert.deepEqual(
    [...buildBinaryAttributes(
      {
        count: 3,
        x: new Float32Array([0, 1, 2]),
        y: new Float32Array([0, 1, 2]),
        type: new Uint16Array([1, 2, 3]),
      },
      [...categories, {code: 3, label: "Unannotated", color: "#778899"}],
      selected,
    ).radii],
    [2.25, 2.25, 0],
  );
});

test("binary filters use zero radius and coordinate direction is reversible", () => {
  const up = buildBinaryAttributes(points, categories, new Set([1, 9]), "up");
  const down = buildBinaryAttributes(points, categories, new Set([1, 9]), "down");
  assert.deepEqual([...up.radii], [2.25, 0, 2.25]);
  assert.deepEqual([...up.positions], [0, 3, 10, 5, 20, 7]);
  assert.deepEqual([...down.positions], [0, -3, 10, -5, 20, -7]);
  assert.deepEqual([...up.colors.slice(0, 4)], [17, 34, 51, 230]);
});

test("hover text includes confidence only for inferred annotations", () => {
  assert.equal(formatHoverText(points, 0, categories, "source"), "T cell");
  assert.equal(formatHoverText(points, 0, categories, "inferred"), "T cell · confidence 0.900");
  assert.equal(formatHoverText(points, 2, categories, "inferred"), "Unknown category 9");
  assert.equal(formatHoverText(points, -1, categories, "inferred"), "");
});

test("reset view centers finite bounds and preserves equal x/y scale", () => {
  const bounds = coordinateBounds(new Float32Array([0, -5, 20, 5]));
  const state = resetViewState(bounds, 1000, 500);
  assert.deepEqual(state.target, [10, 0, 0]);
  assert.equal(state.zoom, Math.log2(45));
});
