import assert from "node:assert/strict";
import test from "node:test";

import {
  decodePointData,
  POINT_DATA_FLAG_CONFIDENCE,
  POINT_DATA_HEADER_BYTES,
} from "../src/point_data.js";

const encoder = new TextEncoder();

function encode({x, y, confidence = null, type, flags = confidence ? POINT_DATA_FLAG_CONFIDENCE : 0}) {
  const count = x.length;
  const size = POINT_DATA_HEADER_BYTES + count * (confidence ? 14 : 10);
  const buffer = new ArrayBuffer(size);
  const bytes = new Uint8Array(buffer);
  bytes.set(encoder.encode("ISCDCCT\0"), 0);
  const view = new DataView(buffer);
  view.setUint16(8, 1, true);
  view.setUint16(10, flags, true);
  view.setUint32(12, count, true);
  view.setBigUint64(16, 32n, true);
  view.setBigUint64(24, BigInt(size), true);
  let offset = 32;
  new Float32Array(buffer, offset, count).set(x);
  offset += count * 4;
  new Float32Array(buffer, offset, count).set(y);
  offset += count * 4;
  if (confidence) {
    new Float32Array(buffer, offset, count).set(confidence);
    offset += count * 4;
  }
  new Uint16Array(buffer, offset, count).set(type);
  return buffer;
}

test("decodes inferred v1 SoA payloads without rearranging values", () => {
  const decoded = decodePointData(encode({
    x: [1.5, -2], y: [3, 4.25], confidence: [0.9, 0.125], type: [7, 65535],
  }));
  assert.equal(decoded.count, 2);
  assert.deepEqual([...decoded.x], [1.5, -2]);
  assert.deepEqual([...decoded.y], [3, 4.25]);
  assert.deepEqual([...decoded.confidence], [...new Float32Array([0.9, 0.125])]);
  assert.deepEqual([...decoded.type], [7, 65535]);
});

test("decodes the Python writer's hard-coded one-point confidence golden vector", () => {
  const hex = (
    "4953434443435400" +
    "0100" +
    "0100" +
    "01000000" +
    "2000000000000000" +
    "2e00000000000000" +
    "0000c03f" +
    "000000c0" +
    "0000803e" +
    "0102"
  );
  const bytes = Uint8Array.from(hex.match(/../g), (pair) => Number.parseInt(pair, 16));
  const decoded = decodePointData(bytes);
  assert.equal(decoded.count, 1);
  assert.deepEqual([...decoded.x], [1.5]);
  assert.deepEqual([...decoded.y], [-2]);
  assert.deepEqual([...decoded.confidence], [0.25]);
  assert.deepEqual([...decoded.type], [513]);
});

test("decodes source-label v1 payloads without confidence", () => {
  const decoded = decodePointData(encode({x: [1], y: [2], type: [3]}));
  assert.equal(decoded.confidence, null);
  assert.deepEqual([...decoded.type], [3]);
});

test("rejects invalid magic, flags, offsets, and exact sizes", () => {
  const valid = encode({x: [1], y: [2], type: [3]});
  const badMagic = valid.slice(0);
  new Uint8Array(badMagic)[0] = 0;
  assert.throws(() => decodePointData(badMagic), /magic/);
  const badFlags = valid.slice(0);
  new DataView(badFlags).setUint16(10, 2, true);
  assert.throws(() => decodePointData(badFlags), /flags/);
  const badOffset = valid.slice(0);
  new DataView(badOffset).setBigUint64(16, 31n, true);
  assert.throws(() => decodePointData(badOffset), /offset/);
  assert.throws(() => decodePointData(valid.slice(0, -1)), /size/);
});

test("rejects non-finite coordinates and invalid confidence", () => {
  assert.throws(
    () => decodePointData(encode({x: [Number.NaN], y: [2], type: [3]})),
    /Non-finite coordinate/,
  );
  assert.throws(
    () => decodePointData(encode({x: [1], y: [Infinity], type: [3]})),
    /Non-finite coordinate/,
  );
  for (const invalid of [Number.NaN, -0.01, 1.01, Infinity]) {
    assert.throws(
      () => decodePointData(encode({x: [1], y: [2], confidence: [invalid], type: [3]})),
      /Invalid confidence/,
    );
  }
});

test("decodes odd point counts from an unaligned view", () => {
  const encoded = new Uint8Array(encode({
    x: [1, 2, 3], y: [4, 5, 6], confidence: [0.1, 0.2, 0.3], type: [7, 8, 9],
  }));
  const padded = new Uint8Array(encoded.length + 1);
  padded.set(encoded, 1);
  const decoded = decodePointData(padded.subarray(1));
  assert.deepEqual([...decoded.x], [1, 2, 3]);
  assert.deepEqual([...decoded.type], [7, 8, 9]);
});
