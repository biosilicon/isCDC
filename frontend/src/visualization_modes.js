import {initialSelectedCategoryCodes} from "./visualization_state.js";

/** Identity is (mode, source sample ID), never the sidecar's local sample key. */
export class VisualizationModes {
  constructor(views) {
    if (!Array.isArray(views) || views.length === 0) throw new Error("No visualization views");
    this.views = views.map((view) => ({...view, viewId: view.viewId || view.kind,
      methodFamily: view.methodFamily || view.viewId || view.kind}));
    if (new Set(this.views.map((view) => view.viewId)).size !== views.length) {
      throw new Error("Duplicate visualization view identity");
    }
    this.selections = new Map();
    this.lastFamilyView = new Map();
    this.select(this.views[0].viewId, this.views[0].samples[0].id);
  }

  select(viewId, sampleId = this.sample?.id) {
    const requested = this.lastFamilyView.get(viewId) || viewId;
    const view = this.views.find((candidate) => candidate.viewId === requested)
      || this.views.find((candidate) => candidate.methodFamily === viewId);
    if (!view) throw new Error("Unknown visualization mode");
    const sample = view.samples.find((candidate) => candidate.id === sampleId) || view.samples[0];
    if (!sample) throw new Error("No sample in visualization mode");
    this.view = view;
    this.lastFamilyView.set(view.methodFamily, view.viewId);
    this.sample = sample;
    this.categories = sample.categories || view.categories;
    const key = JSON.stringify([view.viewId, sample.id]);
    if (!this.selections.has(key)) {
      const selected = view.kind === "spatial_domain"
        ? new Set(this.categories.filter((c) => c.code !== 0).map((c) => c.code))
        : initialSelectedCategoryCodes(this.categories);
      this.selections.set(key, selected);
    }
    this.selectedCodes = this.selections.get(key);
  }
}
