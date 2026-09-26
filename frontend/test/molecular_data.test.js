import assert from "node:assert/strict";
import test from "node:test";
import {decodeMolecularArray, molecularAttributes, molecularHover, molecularRange} from "../src/molecular_data.js";

function array(kind, type, values, states = []) {
  const data = new (type === "int64" ? BigInt64Array : Float64Array)(values);
  const buffer = new ArrayBuffer(32 + data.byteLength + states.length);
  const bytes = new Uint8Array(buffer), header = new DataView(buffer);
  bytes.set([73, 83, 67, 68, 67, 77, 79, 0]); header.setUint16(8, 1, true);
  header.setUint16(10, kind, true); header.setUint32(12, kind === 1 ? values.length / 2 : values.length, true);
  bytes.set(new TextEncoder().encode(type), 16); bytes.set(new Uint8Array(data.buffer), 32);
  bytes.set(states, 32 + data.byteLength); return buffer;
}

test("binary molecular values retain Float64 and integers above 2^53", () => {
  const integer = decodeMolecularArray(array(2, "int64", [9007199254740993n], [0]), 2);
  assert.equal(integer.values[0], 9007199254740993n);
  const values = decodeMolecularArray(array(2, "float64", [0.123456789012345, NaN, 0], [0, 2, 1]), 2);
  assert.equal(values.values[0], 0.123456789012345);
  assert.deepEqual([...values.states], [0, 2, 1]);
  const coordinates = decodeMolecularArray(array(1, "float64", [1, 2, 3, 4]), 1);
  assert.deepEqual([...coordinates.x], [1, 2]); assert.deepEqual([...coordinates.y], [3, 4]);
  assert.throws(() => decodeMolecularArray(array(2, "float64", [NaN], [0]), 2));
  assert.throws(() => decodeMolecularArray(array(2, "float64", [0], [3]), 2));
  assert.throws(() => decodeMolecularArray(array(1, "float64", [NaN, 1]), 1));
  assert.throws(() => decodeMolecularArray(array(2, "float64", [1], [0]).slice(0, 35), 2));
  assert.throws(() => decodeMolecularArray(array(2, "float64", [1], [0]), 1));
});

test("zero, missing and non-finite values stay distinct in color and hover", () => {
  const points = {count: 4, x: [1, 2, 3, 4], y: [1, 2, 3, 4],
    values: [0, 10, 0, NaN], states: [0, 0, 1, 2],
    feature: {label: "Gene", id: "ENSG"}, modality: {name: "rna", value_type: "counts"}, sampleId: "A"};
  const range = molecularRange(points, "counts", "log1p");
  assert.deepEqual([range.valid, range.missing, range.invalid, range.min, range.max], [2, 1, 1, 0, 10]);
  assert.equal(range.scale, "log1p");
  const attrs = molecularAttributes(points, range, "down");
  assert.deepEqual([...attrs.sourceIndices], [2, 3, 0, 1]);
  assert.equal(attrs.positions[1], -3);
  assert.notDeepEqual([...attrs.colors.slice(0, 4)], [...attrs.colors.slice(8, 12)]);
  assert.match(molecularHover(points, 2), /Not measured/);
  assert.match(molecularHover(points, 3), /Non-finite/);
  assert.match(molecularHover(points, 0), /\n0$/);
  assert.equal(molecularHover(points, -1), "");
});

test("nonnegative values default to logarithmic colors and allow an explicit linear scale", () => {
  const points = {count: 3, values: [0, 10, 1000], states: [0, 0, 0]};
  assert.equal(molecularRange(points, "counts").scale, "log1p");
  assert.equal(molecularRange(points, "intensity").scale, "log1p");
  assert.equal(molecularRange(points, "counts", "linear").scale, "linear");
});

test("negative, binary, constant, empty and pre-logged scales", () => {
  const points = {count: 2, values: [-2, 4], states: [0, 0]};
  const range = molecularRange(points, "normalized");
  assert.deepEqual([range.min, range.max, range.scale, range.diverging], [-4, 4, "linear", true]);
  points.values = [0, 0];
  assert.equal(molecularRange(points, "counts").constant, true);
  assert.deepEqual([molecularRange(points, "binary").min, molecularRange(points, "binary").max], [0, 1]);
  assert.equal(molecularRange(points, "binary").scale, "linear");
  assert.equal(molecularRange(points, "log_normalized").scale, "linear");
  points.states = [1, 2];
  assert.equal(molecularRange(points, "counts").valid, 0);
  assert.equal(molecularRange(points, "counts").scale, "linear");
});
