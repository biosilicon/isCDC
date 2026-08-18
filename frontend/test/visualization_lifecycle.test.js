import assert from "node:assert/strict";
import test from "node:test";

import {VisualizationLifecycle} from "../src/visualization_lifecycle.js";

const categories = [{code: 1, label: "T cell", color: "#112233"}];
const samples = [
  {key: "first", id: "First", count: 1, url: "/first"},
  {key: "second", id: "Second", count: 1, url: "/second"},
];
const point = {count: 1, type: new Uint16Array([1])};

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return {promise, resolve, reject};
}

function harness(overrides = {}) {
  const calls = {load: [], render: [], reset: 0, destroy: 0, hide: 0, states: []};
  const lifecycle = new VisualizationLifecycle({
    samples,
    initialSampleKey: "first",
    categories,
    supportsWebGL: () => true,
    load: async (sample, signal) => {
      calls.load.push({sample, signal});
      return point;
    },
    renderer: {
      render: (...args) => calls.render.push(args),
      reset: () => { calls.reset += 1; },
      destroy: () => { calls.destroy += 1; },
    },
    onState: (...args) => calls.states.push(args),
    hide: () => { calls.hide += 1; },
    ...overrides,
  });
  return {lifecycle, calls};
}

test("lazy controller does not fetch until intersection starts it", async () => {
  const {lifecycle, calls} = harness();
  assert.equal(calls.load.length, 0);
  await lifecycle.start();
  assert.equal(calls.load.length, 1);
  assert.equal(calls.render.length, 1);
});

test("WebGL2 failure hides without fetching or creating renderer state", async () => {
  const {lifecycle, calls} = harness({supportsWebGL: () => false});
  await lifecycle.start();
  assert.equal(calls.load.length, 0);
  assert.equal(calls.render.length, 0);
  assert.equal(calls.hide, 1);
  assert.equal(calls.destroy, 1);
});

test("sample switching aborts, ignores stale completion, and reuses renderer", async () => {
  const first = deferred();
  const second = deferred();
  const {lifecycle, calls} = harness({
    load: (sample, signal) => {
      calls.load.push({sample, signal});
      return sample.key === "first" ? first.promise : second.promise;
    },
  });
  const firstRun = lifecycle.start();
  const secondRun = lifecycle.switchSample("second");
  assert.equal(calls.load[0].signal.aborted, true);
  second.resolve(point);
  await secondRun;
  first.resolve(point);
  await firstRun;
  assert.equal(calls.render.length, 1);
  assert.equal(calls.render[0][1].key, "second");
  assert.equal(calls.destroy, 0);
});

test("retry transitions an error to ready", async () => {
  let attempt = 0;
  const {lifecycle, calls} = harness({
    load: async () => {
      attempt += 1;
      if (attempt === 1) throw new Error("temporary");
      return point;
    },
  });
  assert.equal(await lifecycle.start(), false);
  assert.deepEqual(calls.states.at(-1), ["error", "The visualization could not be loaded. You can retry without reloading the page."]);
  assert.equal(await lifecycle.retry(), true);
  assert.deepEqual(calls.states.at(-1), ["ready", ""]);
});

test("reset delegates and context loss hides and releases exactly once", () => {
  const {lifecycle, calls} = harness();
  lifecycle.reset();
  assert.equal(calls.reset, 1);
  lifecycle.contextLost();
  lifecycle.contextLost();
  lifecycle.destroy(true);
  assert.equal(calls.destroy, 1);
  assert.equal(calls.hide, 1);
});

test("destroy aborts a pending request and releases exactly once", () => {
  const pending = deferred();
  const {lifecycle, calls} = harness({load: () => pending.promise});
  lifecycle.start();
  const signal = lifecycle.abortController.signal;
  lifecycle.destroy(false);
  lifecycle.destroy(false);
  assert.equal(signal.aborted, true);
  assert.equal(calls.destroy, 1);
});

test("unknown binary category is reported as a load error", async () => {
  const {lifecycle, calls} = harness({
    load: async () => ({count: 1, type: new Uint16Array([99])}),
  });
  assert.equal(await lifecycle.start(), false);
  assert.equal(calls.render.length, 0);
  assert.equal(calls.states.at(-1)[0], "error");
});
