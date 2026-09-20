import {initialSelectedCategoryCodes} from "./visualization_state.js";

/** Identity is (mode, source sample ID), never the sidecar's local sample key. */
export class VisualizationModes {
  constructor(views) {
    if (!Array.isArray(views) || views.length === 0) throw new Error("No visualization views");
    this.views = views;
    this.selections = new Map();
    this.select(views[0].kind, views[0].samples[0].id);
  }

  select(kind, sampleId = this.sample?.id) {
    const view = this.views.find((candidate) => candidate.kind === kind);
    if (!view) throw new Error("Unknown visualization mode");
    const sample = view.samples.find((candidate) => candidate.id === sampleId) || view.samples[0];
    if (!sample) throw new Error("No sample in visualization mode");
    this.view = view;
    this.sample = sample;
    this.categories = sample.categories || view.categories;
    const key = JSON.stringify([kind, sample.id]);
    if (!this.selections.has(key)) {
      const selected = kind === "spatial_domain"
        ? new Set(this.categories.filter((c) => c.code !== 0).map((c) => c.code))
        : initialSelectedCategoryCodes(this.categories);
      this.selections.set(key, selected);
    }
    this.selectedCodes = this.selections.get(key);
  }
}
