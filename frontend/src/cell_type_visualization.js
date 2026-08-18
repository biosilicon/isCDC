import {Deck, OrthographicView} from "@deck.gl/core";
import {ScatterplotLayer} from "@deck.gl/layers";

import {decodePointData} from "./point_data.js";
import {VisualizationLifecycle} from "./visualization_lifecycle.js";
import {
  buildBinaryAttributes,
  buildLegendEntries,
  coordinateBounds,
  formatHoverText,
  resetViewState,
} from "./visualization_state.js";

const ROOT_ID = "cell-type-visualization";
const CONFIG_ID = "cell-type-visualization-config";

function requiredElement(root, selector) {
  const element = root.querySelector(selector);
  if (!element) throw new Error(`Cell type visualization is missing ${selector}`);
  return element;
}

function webgl2Available() {
  try {
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("webgl2", {failIfMajorPerformanceCaveat: true});
    const available = Boolean(context);
    context?.getExtension("WEBGL_lose_context")?.loseContext();
    return available;
  } catch {
    return false;
  }
}

function validateConfig(value) {
  if (!value || !Array.isArray(value.categories) || !Array.isArray(value.samples)) {
    throw new Error("Invalid cell type visualization configuration");
  }
  if (value.samples.length === 0) throw new Error("Visualization has no samples");
  return value;
}

class CellTypeVisualization {
  constructor(root, config) {
    this.root = root;
    this.config = config;
    this.stage = requiredElement(root, "[data-cell-type-stage]");
    this.canvasHost = requiredElement(root, "[data-cell-type-canvas]");
    this.sampleSelect = requiredElement(root, "[data-cell-type-sample]");
    this.resetButton = requiredElement(root, "[data-cell-type-reset]");
    this.retryButton = requiredElement(root, "[data-cell-type-retry]");
    this.closeButton = requiredElement(root, "[data-cell-type-close]");
    this.status = requiredElement(root, "[data-cell-type-status]");
    this.legend = requiredElement(root, "[data-cell-type-legend]");
    this.tooltip = requiredElement(root, "[data-cell-type-tooltip]");
    this.deck = null;
    this.points = null;
    this.attributes = null;
    this.initialViewState = null;
    this.viewState = null;
    this.selectedCodes = new Set(config.categories.map((category) => Number(category.code)));
    this.currentSampleKey = config.initialSampleKey || config.samples[0].key;
    this.onContextLost = (event) => {
      event.preventDefault();
      this.lifecycle.contextLost();
    };
    this.onResize = () => this.deck?.redraw(true);
    this.onSampleChange = () => {
      this.currentSampleKey = this.sampleSelect.value;
      this.tooltip.hidden = true;
      this.lifecycle.switchSample(this.currentSampleKey);
    };
    this.onReset = () => this.lifecycle.reset();
    this.onRetry = () => this.lifecycle.retry();
    this.onClose = () => this.lifecycle.destroy(true);
    this.onPageHide = () => this.lifecycle.destroy(false);
    this.lifecycle = new VisualizationLifecycle({
      samples: config.samples,
      initialSampleKey: this.currentSampleKey,
      categories: config.categories,
      supportsWebGL: webgl2Available,
      load: (sample, signal) => this.fetchPoints(sample, signal),
      renderer: {
        render: (points, sample) => this.renderPoints(points, sample),
        reset: () => this.resetView(),
        destroy: () => this.destroyRenderer(),
      },
      onState: (state, message) => this.setStatus(state, message),
      hide: () => { this.root.hidden = true; },
    });
    this.bindControls();
    this.populateSamples();
    this.renderLegend(buildLegendEntries(new Uint16Array(), config.categories));
  }

  bindControls() {
    this.sampleSelect.addEventListener("change", this.onSampleChange);
    this.resetButton.addEventListener("click", this.onReset);
    this.retryButton.addEventListener("click", this.onRetry);
    this.closeButton.addEventListener("click", this.onClose);
    window.addEventListener("pagehide", this.onPageHide, {once: true});
  }

  populateSamples() {
    this.sampleSelect.replaceChildren();
    for (const sample of this.config.samples) {
      const option = document.createElement("option");
      option.value = sample.key;
      option.textContent = `${sample.id} (${Number(sample.count).toLocaleString()} points)`;
      option.selected = sample.key === this.currentSampleKey;
      this.sampleSelect.append(option);
    }
  }

  setStatus(state, message) {
    this.root.dataset.visualizationState = state;
    this.status.textContent = message;
    this.status.hidden = !message;
    this.retryButton.hidden = state !== "error";
  }

  start() {
    return this.lifecycle.start();
  }

  async fetchPoints(sample, signal) {
    const response = await fetch(sample.url, {
      signal,
      credentials: "same-origin",
      headers: {Accept: "application/vnd.iscdc.cell-type-points"},
    });
    if (!response.ok) throw new Error(`Point request failed with HTTP ${response.status}`);
    return decodePointData(await response.arrayBuffer());
  }

  renderPoints(points, sample) {
    this.currentSampleKey = sample.key;
    this.sampleSelect.value = sample.key;
    this.points = points;
    this.renderLegend(buildLegendEntries(points.type, this.config.categories));
    this.rebuildLayer(true);
  }

  rebuildLayer(resetView = false) {
    if (!this.points) return;
    this.attributes = buildBinaryAttributes(
      this.points,
      this.config.categories,
      this.selectedCodes,
      this.config.yAxis,
    );
    const data = {
      length: this.points.count,
      attributes: {
        getPosition: {value: this.attributes.positions, size: 2},
        getFillColor: {value: this.attributes.colors, size: 4},
        getRadius: {value: this.attributes.radii, size: 1},
      },
    };
    if (resetView || !this.initialViewState) {
      const rect = this.canvasHost.getBoundingClientRect();
      this.initialViewState = resetViewState(
        coordinateBounds(this.attributes.positions),
        rect.width,
        rect.height,
      );
      this.viewState = this.initialViewState;
    }
    const layer = new ScatterplotLayer({
      id: `cell-types-${this.currentSampleKey}`,
      data,
      pickable: true,
      filled: true,
      stroked: false,
      radiusUnits: "pixels",
      radiusMinPixels: 0,
      radiusMaxPixels: 8,
      onHover: (info) => this.showTooltip(info),
    });
    const props = {
      views: [new OrthographicView({id: "cell-types", controller: true})],
      viewState: this.viewState,
      controller: {dragPan: true, scrollZoom: true, doubleClickZoom: true, touchZoom: true},
      layers: [layer],
      onViewStateChange: ({viewState}) => {
        this.viewState = viewState;
        this.deck?.setProps({viewState});
      },
      getCursor: ({isDragging, isHovering}) => isDragging ? "grabbing" : isHovering ? "pointer" : "grab",
    };
    if (this.deck) {
      this.deck.setProps(props);
    } else {
      this.deck = new Deck({parent: this.canvasHost, width: "100%", height: "100%", ...props});
      this.deck.getCanvas()?.addEventListener("webglcontextlost", this.onContextLost, {once: true});
      window.addEventListener("resize", this.onResize, {passive: true});
    }
  }

  resetView() {
    if (!this.points || !this.attributes) return;
    const rect = this.canvasHost.getBoundingClientRect();
    this.initialViewState = resetViewState(
      coordinateBounds(this.attributes.positions),
      rect.width,
      rect.height,
    );
    this.viewState = this.initialViewState;
    this.deck?.setProps({viewState: this.viewState});
  }

  showTooltip(info) {
    const text = formatHoverText(
      this.points,
      info.index,
      this.config.categories,
      this.config.annotationKind,
    );
    if (!text) {
      this.tooltip.hidden = true;
      return;
    }
    this.tooltip.textContent = text;
    this.tooltip.style.left = `${info.x + 12}px`;
    this.tooltip.style.top = `${info.y + 12}px`;
    this.tooltip.hidden = false;
  }

  renderLegend(entries) {
    this.legend.replaceChildren();
    const controls = document.createElement("div");
    controls.className = "cell-type-legend-controls";
    const selectAll = document.createElement("button");
    selectAll.type = "button";
    selectAll.className = "btn btn-sm btn-link";
    selectAll.dataset.cellTypeSelectAll = "";
    selectAll.textContent = "Select all";
    const clearAll = document.createElement("button");
    clearAll.type = "button";
    clearAll.className = "btn btn-sm btn-link";
    clearAll.dataset.cellTypeClearAll = "";
    clearAll.textContent = "Clear all";
    controls.append(selectAll, clearAll);
    this.legend.append(controls);

    const list = document.createElement("div");
    list.className = "cell-type-legend-list";
    for (const entry of entries) {
      const label = document.createElement("label");
      label.className = "cell-type-legend-item";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = this.selectedCodes.has(entry.code);
      checkbox.value = String(entry.code);
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) this.selectedCodes.add(entry.code);
        else this.selectedCodes.delete(entry.code);
        this.rebuildLayer(false);
      });
      const swatch = document.createElement("span");
      swatch.className = "cell-type-legend-swatch";
      const [red, green, blue, alpha] = entry.color;
      swatch.style.backgroundColor = `rgba(${red}, ${green}, ${blue}, ${alpha / 255})`;
      const text = document.createElement("span");
      text.textContent = entry.label;
      if (entry.cellOntologyId) text.title = entry.cellOntologyId;
      const count = document.createElement("span");
      count.textContent = `${entry.count.toLocaleString()} (${(entry.proportion * 100).toFixed(1)}%)`;
      label.append(checkbox, swatch, text, count);
      list.append(label);
    }
    this.legend.append(list);
    const setAll = (selected) => {
      this.selectedCodes = selected ? new Set(entries.map((entry) => entry.code)) : new Set();
      for (const checkbox of list.querySelectorAll('input[type="checkbox"]')) checkbox.checked = selected;
      this.rebuildLayer(false);
    };
    selectAll.addEventListener("click", () => setAll(true));
    clearAll.addEventListener("click", () => setAll(false));
  }

  destroyRenderer() {
    this.sampleSelect.removeEventListener("change", this.onSampleChange);
    this.resetButton.removeEventListener("click", this.onReset);
    this.retryButton.removeEventListener("click", this.onRetry);
    this.closeButton.removeEventListener("click", this.onClose);
    window.removeEventListener("pagehide", this.onPageHide);
    window.removeEventListener("resize", this.onResize);
    this.deck?.getCanvas()?.removeEventListener("webglcontextlost", this.onContextLost);
    this.deck?.finalize();
    this.deck = null;
    this.points = null;
    this.attributes = null;
    this.tooltip.hidden = true;
  }
}

function initialize() {
  const root = document.getElementById(ROOT_ID);
  const configElement = document.getElementById(CONFIG_ID);
  if (!root || !configElement) return;
  let visualization;
  try {
    const config = validateConfig(JSON.parse(configElement.textContent));
    visualization = new CellTypeVisualization(root, config);
  } catch {
    root.hidden = true;
    return;
  }

  if (!("IntersectionObserver" in window)) {
    visualization.start();
    return;
  }
  const observer = new IntersectionObserver((entries) => {
    if (!entries.some((entry) => entry.isIntersecting)) return;
    observer.disconnect();
    visualization.start();
  }, {rootMargin: "240px 0px"});
  observer.observe(root);
  visualization.closeButton.addEventListener("click", () => observer.disconnect(), {once: true});
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initialize, {once: true});
} else {
  initialize();
}
