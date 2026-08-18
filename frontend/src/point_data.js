const MAGIC = Uint8Array.of(73, 83, 67, 68, 67, 67, 84, 0); // ISCDCCT\0

export const POINT_DATA_VERSION = 1;
export const POINT_DATA_HEADER_BYTES = 32;
export const POINT_DATA_FLAG_CONFIDENCE = 0x0001;
export const POINT_DATA_KNOWN_FLAGS = POINT_DATA_FLAG_CONFIDENCE;

function asBytes(input) {
  if (input instanceof ArrayBuffer) {
    return new Uint8Array(input);
  }
  if (ArrayBuffer.isView(input)) {
    return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  }
  throw new TypeError("Point data must be an ArrayBuffer or typed-array view");
}

function validateMagic(bytes) {
  for (let index = 0; index < MAGIC.length; index += 1) {
    if (bytes[index] !== MAGIC[index]) {
      throw new Error("Invalid cell type point-data magic");
    }
  }
}

/** Decode the strict little-endian cell type point format without copying payloads. */
export function decodePointData(input) {
  let bytes = asBytes(input);
  if (bytes.byteLength < POINT_DATA_HEADER_BYTES) {
    throw new Error("Cell type point data is shorter than its 32-byte header");
  }
  validateMagic(bytes);

  let view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = view.getUint16(8, true);
  const flags = view.getUint16(10, true);
  const count = view.getUint32(12, true);
  const payloadOffset = view.getBigUint64(16, true);
  const declaredSize = view.getBigUint64(24, true);

  if (version !== POINT_DATA_VERSION) {
    throw new Error(`Unsupported cell type point-data version: ${version}`);
  }
  if ((flags & ~POINT_DATA_KNOWN_FLAGS) !== 0) {
    throw new Error(`Unsupported cell type point-data flags: 0x${flags.toString(16)}`);
  }
  if (payloadOffset !== BigInt(POINT_DATA_HEADER_BYTES)) {
    throw new Error(`Invalid cell type point-data payload offset: ${payloadOffset}`);
  }

  const hasConfidence = (flags & POINT_DATA_FLAG_CONFIDENCE) !== 0;
  const bytesPerPoint = hasConfidence ? 14 : 10;
  const expectedSize = POINT_DATA_HEADER_BYTES + bytesPerPoint * count;
  if (declaredSize !== BigInt(expectedSize) || bytes.byteLength !== expectedSize) {
    throw new Error(
      `Invalid cell type point-data size: expected ${expectedSize}, received ${bytes.byteLength}`,
    );
  }

  // A caller may provide an unaligned Uint8Array slice. Copy only in that unusual case.
  if ((bytes.byteOffset + POINT_DATA_HEADER_BYTES) % Float32Array.BYTES_PER_ELEMENT !== 0) {
    bytes = Uint8Array.from(bytes);
    view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }

  const start = bytes.byteOffset + POINT_DATA_HEADER_BYTES;
  const floatBytes = count * Float32Array.BYTES_PER_ELEMENT;
  const x = new Float32Array(bytes.buffer, start, count);
  const y = new Float32Array(bytes.buffer, start + floatBytes, count);
  let confidence = null;
  let typeOffset = start + 2 * floatBytes;
  if (hasConfidence) {
    confidence = new Float32Array(bytes.buffer, typeOffset, count);
    typeOffset += floatBytes;
  }
  const type = new Uint16Array(bytes.buffer, typeOffset, count);

  for (let index = 0; index < count; index += 1) {
    if (!Number.isFinite(x[index]) || !Number.isFinite(y[index])) {
      throw new Error(`Non-finite coordinate at point ${index}`);
    }
    if (
      confidence !== null
      && (!Number.isFinite(confidence[index]) || confidence[index] < 0 || confidence[index] > 1)
    ) {
      throw new Error(`Invalid confidence at point ${index}`);
    }
  }

  return {version, flags, count, x, y, confidence, type};
}
