import {createRequestGate, validateCategoryCodes} from "./visualization_state.js";

/**
 * Coordinates lazy startup, racing fetches, and resource ownership independently of the DOM.
 * The renderer is deliberately injected so these transitions stay unit-testable.
 */
export class VisualizationLifecycle {
  constructor({samples, initialSampleKey, categories, supportsWebGL, load, renderer, onState, hide}) {
    this.samples = samples;
    this.categories = categories;
    this.supportsWebGL = supportsWebGL;
    this.load = load;
    this.renderer = renderer;
    this.onState = onState;
    this.hide = hide;
    this.currentSampleKey = initialSampleKey || samples[0]?.key;
    this.requestGate = createRequestGate();
    this.abortController = null;
    this.started = false;
    this.destroyed = false;
    this.rendererDestroyed = false;
  }

  start() {
    if (this.started || this.destroyed) return Promise.resolve(false);
    this.started = true;
    if (!this.supportsWebGL()) {
      this.destroy(true);
      return Promise.resolve(false);
    }
    return this.switchSample(this.currentSampleKey);
  }

  async switchSample(sampleKey) {
    if (this.destroyed) return false;
    const sample = this.samples.find((candidate) => candidate.key === sampleKey);
    if (!sample) {
      this.onState("error", "The selected visualization sample is unavailable.");
      return false;
    }
    this.currentSampleKey = sampleKey;
    this.abortController?.abort();
    this.abortController = new AbortController();
    const token = this.requestGate.begin();
    this.onState("loading", `Loading ${sample.id}…`);
    try {
      const points = await this.load(sample, this.abortController.signal);
      if (!this.requestGate.isCurrent(token) || this.destroyed) return false;
      if (points.count !== Number(sample.count)) throw new Error(`Point count mismatch for ${sample.id}`);
      validateCategoryCodes(points.type, this.categories);
      this.renderer.render(points, sample);
      this.onState("ready", "");
      return true;
    } catch (error) {
      if (!this.requestGate.isCurrent(token) || this.destroyed || error?.name === "AbortError") {
        return false;
      }
      this.onState("error", "The visualization could not be loaded. You can retry without reloading the page.");
      return false;
    }
  }

  retry() {
    return this.switchSample(this.currentSampleKey);
  }

  reset() {
    if (!this.destroyed) this.renderer.reset();
  }

  contextLost() {
    this.destroy(true);
  }

  destroy(hideRegion = false) {
    if (this.destroyed) return;
    this.destroyed = true;
    this.requestGate.invalidate();
    this.abortController?.abort();
    this.abortController = null;
    if (!this.rendererDestroyed) {
      this.rendererDestroyed = true;
      this.renderer.destroy();
    }
    if (hideRegion) this.hide();
  }
}
