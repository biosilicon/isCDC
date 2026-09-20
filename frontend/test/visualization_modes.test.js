import assert from "node:assert/strict";
import test from "node:test";
import {VisualizationModes} from "../src/visualization_modes.js";
import {buildBinaryAttributes} from "../src/visualization_state.js";
import {decodePointData} from "../src/point_data.js";

const cell = {kind: "cell_type", categories: [{code: 0, label: "Unannotated"}, {code: 1, label: "T cell"}, {code: 2, label: "Uncertain"}],
  samples: [{id: "A", key: "cell_a"}, {id: "B", key: "cell_b"}]};
const domains = {kind: "spatial_domain", samples: [
  {id: "A", key: "domain_a", categories: [{code: 0, label: "Not analyzed"}, {code: 1, label: "Domain 1"}]},
  {id: "B", key: "domain_b", categories: [{code: 1, label: "Domain 1"}, {code: 2, label: "Domain 2"}]},
]};

test("mode changes retain sample identity and independent selections", () => {
  const modes = new VisualizationModes([cell, domains]);
  assert.deepEqual([...modes.selectedCodes], [1]);
  modes.selectedCodes.add(0);
  modes.selectedCodes.add(2);
  modes.select("spatial_domain");
  assert.equal(modes.sample.key, "domain_a");
  assert.deepEqual([...modes.selectedCodes], [1]);
  modes.selectedCodes.clear();
  modes.select("spatial_domain", "B");
  assert.deepEqual([...modes.selectedCodes], [1, 2]);
  modes.select("cell_type", "A");
  assert.deepEqual([...modes.selectedCodes], [1, 0, 2]);
  modes.select("cell_type", "B");
  assert.deepEqual([...modes.selectedCodes], [1]);
  modes.select("spatial_domain", "A");
  assert.deepEqual([...modes.selectedCodes], []);
});

test("domain binary discriminator and 700k point attributes", () => {
  const count = 700000;
  const bytes = new Uint8Array(32 + 10 * count);
  bytes.set([73,83,67,68,67,83,68,0]);
  const header = new DataView(bytes.buffer);
  header.setUint16(8, 1, true);
  header.setUint32(12, count, true);
  header.setBigUint64(16, 32n, true);
  header.setBigUint64(24, BigInt(bytes.length), true);
  const labels = new Uint16Array(bytes.buffer, 32 + 8 * count, count);
  labels.fill(1);
  labels[0] = 0;
  assert.throws(() => decodePointData(bytes), /magic/);
  const points = decodePointData(bytes, "spatial_domain");
  const modes = new VisualizationModes([domains]);
  const attributes = buildBinaryAttributes(points, modes.categories, modes.selectedCodes, "up");
  assert.equal(attributes.radii.length, count);
  assert.equal(attributes.radii[0], 0);
  assert.equal(attributes.radii[count - 1], 2.25);
});
