import {expect, test} from "@playwright/test";

const datasetPath = process.env.ISCDC_MOLECULAR_PAGE || "/databases/xenium_human_ccrcc_ffpe_rna_protein";

async function requirePublishedWebGL(page) {
  await expect(page.locator('[data-visualization-mode="molecular"]')).toBeEnabled();
  const available = await page.evaluate(() => {
    const gl = document.createElement("canvas").getContext("webgl2", {failIfMajorPerformanceCaveat: true});
    gl?.getExtension("WEBGL_lose_context")?.loseContext();
    return Boolean(gl);
  });
  test.skip(!available, "This browser host has no WebGL2; the separate fallback test covers this state.");
}

test("molecular feature search, scales, mode transitions and responsive layout", async ({page}, testInfo) => {
  test.setTimeout(300_000);
  const largeSampleExpect = expect.configure({timeout: 90_000});
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => { if (message.type() === "error") errors.push(message.text()); });
  await page.goto(datasetPath);
  await requirePublishedWebGL(page);
  const region = page.locator("#cell-type-visualization");
  await region.scrollIntoViewIfNeeded();
  await region.locator('[data-visualization-mode="molecular"]').click();
  await expect(region).toHaveAttribute("data-visualization-state", "ready", {timeout: 90_000});
  await expect(region.locator("canvas")).toBeVisible();
  await expect(region.locator(".molecular-colorbar")).toBeVisible();
  const search = region.locator("[data-molecular-search]");
  await search.focus();
  await expect(region.locator(".molecular-result").first()).toBeVisible();
  await region.locator(".molecular-result").first().focus();
  await page.keyboard.press("Escape");
  await expect(region.locator("[data-molecular-results]")).toBeHidden();
  await search.fill("this_feature_does_not_exist_000");
  await expect(region.locator("[data-molecular-search-status]")).toContainText("No matching");
  await search.fill("ENSG");
  await expect(region.locator(".molecular-result").first()).toBeVisible();
  const chosen = await region.locator(".molecular-result").nth(1).textContent();
  await region.locator(".molecular-result").nth(1).click();
  await expect(region.locator("[data-molecular-selected]")).toContainText(chosen.split(" · ")[0]);
  await largeSampleExpect(region).toHaveAttribute("data-visualization-state", "ready");
  await region.locator("[data-molecular-scale]").selectOption("log1p");
  await expect(region.locator("[data-cell-type-legend]")).toContainText("Log1p colors");
  await region.locator("[data-molecular-modality]").selectOption("protein");
  await largeSampleExpect(region).toHaveAttribute("data-visualization-state", "ready");
  await region.locator("[data-molecular-modality]").selectOption("rna");
  await largeSampleExpect(region).toHaveAttribute("data-visualization-state", "ready");
  await expect(region.locator("[data-molecular-selected]")).toContainText(chosen.split(" · ")[0]);
  const cellButton = region.locator('[data-visualization-mode="cell_type"]');
  if (await cellButton.isEnabled()) {
    await cellButton.click();
    await expect(region).toHaveAttribute("data-visualization-state", "ready", {timeout: 90_000});
    await expect(region.locator("[data-molecular-controls]")).toBeHidden();
    for (const checkbox of await region.locator(".cell-type-legend-item")
      .filter({has: page.locator("span", {hasText: /^(Unannotated|Uncertain)$/})})
      .locator('input[type="checkbox"]').all()) await expect(checkbox).not.toBeChecked();
    await region.locator('[data-visualization-mode="molecular"]').click();
    await largeSampleExpect(region).toHaveAttribute("data-visualization-state", "ready");
  }
  await region.locator("[data-cell-type-reset]").click();
  const visualIssues = await region.evaluate((element) => {
    const issues = [];
    for (const node of element.querySelectorAll("button,input,select")) {
      const box = node.getBoundingClientRect();
      if (!box.width || !box.height) continue;
      if (parseFloat(getComputedStyle(node).fontSize) < 12) issues.push("tiny control text");
      if (box.right > document.documentElement.clientWidth + 1) issues.push("control overflow");
    }
    return issues;
  });
  expect(visualIssues).toEqual([]);
  await region.screenshot({path: testInfo.outputPath("molecular-desktop.png")});
  await page.setViewportSize({width: 390, height: 844});
  await region.scrollIntoViewIfNeeded();
  await expect(region.locator("[data-molecular-modality]")).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
  expect(overflow).toBe(false);
  await region.screenshot({path: testInfo.outputPath("molecular-mobile.png")});
  expect(errors).toEqual([]);
});

test("logarithmic defaults, manual linear colors and binary ATAC fallback", async ({page}, testInfo) => {
  test.setTimeout(90_000);
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/databases/GSE279771_human_gbm_space_seq_atac_rna");
  await requirePublishedWebGL(page);
  const region = page.locator("#cell-type-visualization");
  await region.locator('[data-visualization-mode="molecular"]').click();
  await expect(region).toHaveAttribute("data-visualization-state", "ready", {timeout: 60_000});
  const scale = region.locator("[data-molecular-scale]");
  await expect(scale).toHaveValue("log1p");
  await expect(region.locator("[data-cell-type-legend]")).toContainText("Log1p colors");
  const search = region.locator("[data-molecular-search]");
  for (const zeroFeature of ["FAM138A", "ENSG00000237613"]) {
    await search.fill(zeroFeature);
    await expect(region.locator("[data-molecular-search-status]")).toContainText("No matching features");
    await expect(region.locator(".molecular-result")).toHaveCount(0);
  }
  await scale.selectOption("linear");
  await expect(region.locator("[data-cell-type-legend]")).toContainText("Linear colors");
  await search.fill("");
  await region.locator(".molecular-result").nth(1).click();
  await expect(region).toHaveAttribute("data-visualization-state", "ready");
  await expect(scale).toHaveValue("linear");
  await region.locator("[data-molecular-modality]").selectOption("atac");
  await expect(region).toHaveAttribute("data-visualization-state", "ready", {timeout: 60_000});
  await expect(scale).toHaveValue("linear");
  await expect(region.locator('[data-molecular-scale] option[value="log1p"]')).toHaveJSProperty("disabled", true);
  await region.locator("[data-molecular-search]").fill("chr1:10000-15000");
  await expect(region.locator("[data-molecular-search-status]")).toContainText("No matching features");
  await expect(region.locator(".molecular-result")).toHaveCount(0);
  await search.fill("chr1:");
  const activeInterval = region.locator(".molecular-result").first();
  await expect(activeInterval).toBeVisible();
  const interval = await activeInterval.textContent();
  await activeInterval.click();
  await expect(region.locator("[data-molecular-selected]")).toContainText(interval);
  await expect(region).toHaveAttribute("data-visualization-state", "ready");
  await expect(region.locator("[data-cell-type-legend]")).toContainText("binary");
  await expect(region.locator(".molecular-colorbar-ticks")).toHaveText("01");
  await region.locator("[data-molecular-modality]").selectOption("rna");
  await expect(region).toHaveAttribute("data-visualization-state", "ready");
  await expect(scale).toHaveValue("log1p");
  await expect(region.locator("[data-cell-type-legend]")).toContainText("Log1p colors");
  await region.screenshot({path: testInfo.outputPath("logarithmic-desktop.png")});
  await page.setViewportSize({width: 390, height: 844});
  await region.scrollIntoViewIfNeeded();
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)).toBe(false);
  await region.screenshot({path: testInfo.outputPath("logarithmic-mobile.png")});
  expect(errors).toEqual([]);
});

test("unavailable WebGL leaves metadata usable without requesting molecular arrays", async ({page}) => {
  const errors = [], arrays = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("request", (request) => { if (request.url().includes("/molecular-visualization/")) arrays.push(request.url()); });
  await page.addInitScript(() => {
    const original = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function(type, ...args) {
      return type === "webgl2" ? null : original.call(this, type, ...args);
    };
  });
  await page.goto(datasetPath);
  const region = page.locator("#cell-type-visualization");
  await expect(page.locator("#cell-type-visualization-config")).toHaveCount(1);
  await region.evaluate((element) => element.scrollIntoView());
  await expect(region).toBeHidden();
  await expect(page.getByRole("heading", {name: "File metadata", exact: true})).toBeVisible();
  expect(arrays).toEqual([]);
  expect(errors).toEqual([]);
});
