import {decodeMolecularArray, DIVERGING, molecularRange, VIRIDIS} from "./molecular_data.js";

/** Owns continuous-value controls without introducing category/annotation semantics. */
export class MolecularControls {
  constructor(root, view, {onDataChange, onScaleChange}) {
    this.root = root; this.view = view; this.onDataChange = onDataChange; this.onScaleChange = onScaleChange;
    this.controls = root.querySelector("[data-molecular-controls]");
    this.select = root.querySelector("[data-molecular-modality]");
    this.search = root.querySelector("[data-molecular-search]");
    this.results = root.querySelector("[data-molecular-results]");
    this.items = root.querySelector("[data-molecular-result-items]");
    this.status = root.querySelector("[data-molecular-search-status]");
    this.more = root.querySelector("[data-molecular-more]");
    this.selectedLabel = root.querySelector("[data-molecular-selected]");
    this.scaleSelect = root.querySelector("[data-molecular-scale]");
    this.requestedScale = "log1p";
    this.scaleSelect.value = this.requestedScale;
    this.selections = new Map(view.modalities.map((modality) => [modality.name, modality.first_feature]));
    this.modality = view.modalities.find((m) => m.name === "rna") || view.modalities[0];
    this.searchSequence = 0;
    this.active = false;
    this.events = new AbortController();
    const options = {signal: this.events.signal};
    for (const modality of view.modalities) {
      const option = document.createElement("option");
      const count = modality.n_searchable ?? modality.n_vars;
      option.value = modality.name; option.textContent = `${modality.name} · ${count.toLocaleString()} searchable features`;
      this.select.append(option);
    }
    this.select.value = this.modality.name;
    this.select.addEventListener("change", () => {
      this.cancelSearch(); this.results.hidden = true; this.search.value = "";
      this.modality = this.view.modalities.find((m) => m.name === this.select.value);
      this.requestedScale = "log1p";
      this.scaleSelect.value = this.requestedScale; this.updateSelection(); this.onDataChange();
    }, options);
    this.search.addEventListener("input", () => {
      this.cancelSearch();
      this.timer = setTimeout(() => this.searchFeatures(0), 200);
    }, options);
    this.search.addEventListener("focus", () => {
      if (!this.suppressFocusSearch) this.searchFeatures(0);
    }, options);
    this.controls.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        this.cancelSearch(); this.results.hidden = true; this.focusSearchWithoutOpening();
      }
    }, options);
    this.controls.addEventListener("focusout", (event) => {
      if (event.relatedTarget && !this.controls.contains(event.relatedTarget)) this.results.hidden = true;
    }, options);
    document.addEventListener("pointerdown", (event) => {
      if (!this.controls.contains(event.target)) this.results.hidden = true;
    }, options);
    this.more.addEventListener("click", () => this.searchFeatures(this.nextOffset), options);
    this.scaleSelect.addEventListener("change", () => {
      this.requestedScale = this.scaleSelect.value; this.onScaleChange();
    }, options);
    this.updateSelection();
  }

  get feature() { return this.selections.get(this.modality.name); }

  focusSearchWithoutOpening() {
    this.suppressFocusSearch = true; this.search.focus(); this.suppressFocusSearch = false;
  }

  updateSelection() {
    this.selectedLabel.textContent = `${this.feature.label} · ${this.feature.id} · ${this.modality.value_type}`;
    this.selectedLabel.title = this.selectedLabel.textContent;
  }

  setActive(active) {
    this.active = active; this.controls.hidden = !active;
    if (!active) { this.cancelSearch(); this.results.hidden = true; }
  }

  cancelSearch() {
    clearTimeout(this.timer); this.searchAbort?.abort(); this.searchSequence++;
  }

  async searchFeatures(offset) {
    if (!this.active || offset === null) return;
    this.cancelSearch();
    const token = this.searchSequence, modality = this.modality.name;
    this.searchAbort = new AbortController();
    this.results.hidden = false; this.status.textContent = "Searching…"; this.more.hidden = true;
    if (offset === 0) this.items.replaceChildren();
    try {
      const params = new URLSearchParams({q: this.search.value, offset: String(offset), limit: "50"});
      const response = await fetch(`${this.view.baseUrl}/modalities/${encodeURIComponent(modality)}/features?${params}`,
        {signal: this.searchAbort.signal, credentials: "same-origin"});
      if (!response.ok) throw new Error("Feature search failed");
      const result = await response.json();
      if (token !== this.searchSequence || !this.active) return;
      for (const feature of result.items) {
        const button = document.createElement("button");
        button.type = "button"; button.className = "molecular-result";
        button.textContent = feature.id === feature.label ? feature.id : `${feature.label} · ${feature.id}`;
        button.addEventListener("click", () => {
          this.selections.set(modality, feature); this.results.hidden = true;
          this.cancelSearch(); this.updateSelection(); this.search.value = "";
          this.focusSearchWithoutOpening(); this.onDataChange();
        });
        this.items.append(button);
      }
      this.nextOffset = result.nextOffset; this.more.hidden = result.nextOffset === null;
      this.status.textContent = this.items.childElementCount ? `${this.items.childElementCount} results shown`
        : this.modality.n_searchable === 0 ? "All features in this modality have zero values."
          : "No matching features. All-zero features are hidden. Try a source ID or name.";
    } catch (error) {
      if (token === this.searchSequence && error.name !== "AbortError") {
        this.status.textContent = "Search unavailable. Focus the search field or edit your query to retry.";
      }
    }
  }

  async load(sample, signal) {
    const feature = this.feature, modality = this.modality;
    const read = async (url, kind) => {
      const response = await fetch(url, {signal, credentials: "same-origin"});
      if (!response.ok) throw new Error(`Molecular request failed: ${response.status}`);
      return decodeMolecularArray(await response.arrayBuffer(), kind);
    };
    const geometry = this.geometry?.key === sample.key ? this.geometry.data : null;
    const url = `${this.view.baseUrl}/samples/${sample.key}/modalities/${encodeURIComponent(modality.name)}/features/${feature.key}`;
    const [coordinates, vector] = await Promise.all([
      geometry || read(sample.url, 1), read(url, 2),
    ]);
    if (coordinates.count !== sample.count || vector.count !== sample.count) throw new Error("Molecular alignment mismatch");
    if (!signal.aborted) this.geometry = {key: sample.key, data: coordinates};
    return {...coordinates, ...vector, feature, modality, sampleId: sample.id};
  }

  renderLegend(host, points) {
    host.replaceChildren();
    // A temporary linear fallback must not replace the visitor's preferred scale.
    const range = molecularRange(points, points.modality.value_type, this.requestedScale);
    this.scaleSelect.querySelector('[value="log1p"]').disabled = !range.logAllowed;
    this.scaleSelect.value = range.scale;
    const title = document.createElement("p");
    title.className = "small fw-semibold molecular-legend-name"; title.textContent = points.feature.label;
    const id = document.createElement("p"); id.className = "small text-secondary molecular-legend-name";
    id.textContent = points.feature.id;
    const bar = document.createElement("div"); bar.className = "molecular-colorbar";
    const palette = range.diverging ? DIVERGING : VIRIDIS;
    bar.style.background = `linear-gradient(to right, ${palette.map((c) => `rgb(${c.join(",")})`).join(",")})`;
    const ticks = document.createElement("div"); ticks.className = "molecular-colorbar-ticks small";
    for (const value of [range.min, range.max]) {
      const tick = document.createElement("span");
      tick.textContent = Number(value.toPrecision(5)).toLocaleString("en", {maximumSignificantDigits: 5});
      ticks.append(tick);
    }
    const details = document.createElement("p"); details.className = "small text-secondary mt-2 mb-2";
    details.textContent = `${points.modality.value_type} · ${range.scale === "log1p" ? "Log1p colors" : "Linear colors"}. Range within this sample.`;
    const counts = document.createElement("p"); counts.className = "small molecular-counts mb-0";
    counts.textContent = `${range.valid.toLocaleString()} measured\n${range.missing.toLocaleString()} not measured\n${range.invalid.toLocaleString()} non-finite`;
    host.append(title, id, bar, ticks, details, counts);
    if (!range.valid || range.constant) {
      const note = document.createElement("p"); note.className = "small mt-2 mb-0";
      note.textContent = !range.valid ? "No finite measurements in this sample."
        : `All measured values are ${String(points.values[points.states.findIndex((s) => s === 0)])}.`;
      host.append(note);
    }
    return range;
  }

  destroy() { this.cancelSearch(); this.events.abort(); this.geometry = null; }
}
