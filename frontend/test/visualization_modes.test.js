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

test("molecular mode preserves sample identity and categorical selections", () => {
  const molecular = {kind: "molecular", categories: [], samples: [{id: "A", key: "s0"}, {id: "B", key: "s1"}]};
  const modes = new VisualizationModes([cell, molecular]);
  modes.select("cell_type", "B"); modes.selectedCodes.add(2);
  modes.select("molecular");
  assert.equal(modes.sample.key, "s1"); assert.equal(modes.selectedCodes.size, 0);
  modes.select("cell_type");
  assert.equal(modes.sample.id, "B"); assert.deepEqual([...modes.selectedCodes], [1, 2]);
});

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

test("RNA and SpatialGLUE share point type but preserve independent view state", () => {
  const glue = {...domains, viewId: "spatialglue"};
  const modes = new VisualizationModes([cell, domains, glue]);
  modes.select("spatial_domain", "B");
  modes.selectedCodes.delete(1);
  modes.select("spatialglue");
  assert.equal(modes.view.kind, "spatial_domain");
  assert.equal(modes.sample.id, "B");
  assert.deepEqual([...modes.selectedCodes], [1, 2]);
  modes.selectedCodes.delete(2);
  modes.select("spatial_domain");
  assert.deepEqual([...modes.selectedCodes], [2]);
  modes.select("spatialglue");
  assert.deepEqual([...modes.selectedCodes], [1]);
  assert.throws(() => new VisualizationModes([domains, domains]), /Duplicate/);
});

test("four modality combinations retain their legend and family selection", () => {
  const combinations = ["rna__protein__atac", "rna__protein__histone", "rna__atac__histone", "protein__atac__histone"];
  const modes = new VisualizationModes([cell, domains, ...combinations.map((id) => ({
    ...domains, viewId: `spatialglue:${id}`, methodFamily: "spatialglue",
  }))]);
  modes.select("spatialglue", "B");
  assert.equal(modes.view.viewId, "spatialglue:rna__protein__atac");
  modes.selectedCodes.clear();
  modes.select("spatialglue:protein__atac__histone");
  assert.deepEqual([...modes.selectedCodes], [1, 2]);
  modes.selectedCodes.delete(1);
  modes.select("cell_type");
  modes.select("spatialglue");
  assert.equal(modes.sample.id, "B");
  assert.equal(modes.view.viewId, "spatialglue:protein__atac__histone");
  assert.deepEqual([...modes.selectedCodes], [2]);
  modes.select("spatialglue:rna__protein__atac");
  assert.equal(modes.selectedCodes.size, 0);
});
