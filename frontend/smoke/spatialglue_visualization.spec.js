import {expect, test} from "@playwright/test";

const databasePath = process.env.ISCDC_GLUE_DATABASE_PATH;
test("SpatialGLUE keeps camera, sample, independent legend and method details", async ({page}, testInfo) => {
  test.skip(!databasePath, "Set ISCDC_GLUE_DATABASE_PATH to a Database with RNA and SpatialGLUE sidecars");
  test.setTimeout(120000);
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => { if (message.type() === "error") errors.push(message.text()); });
  await page.route("**/favicon.ico", (route) => route.fulfill({status: 204}));
  await page.goto(databasePath);
  const region = page.locator("#cell-type-visualization");
  await region.scrollIntoViewIfNeeded();
  await expect(region).toHaveAttribute("data-visualization-state", "ready", {timeout: 90000});
  const rna = region.getByRole("button", {name: "RNA domains", exact: true});
  const glue = region.getByRole("button", {name: "SpatialGLUE domains", exact: true});
  await rna.click();
  await expect(region).toHaveAttribute("data-visualization-state", "ready");
  const canvas = await region.locator("canvas").elementHandle();
  const sample = await region.locator("[data-cell-type-sample]").inputValue();
  const domains = region.locator(".cell-type-legend-item").filter({hasText: /^Domain /});
  await domains.first().locator("input").uncheck();
  const screenshotPlot = async (name) => {
    await region.locator("canvas").scrollIntoViewIfNeeded();
    const box = await region.locator("canvas").boundingBox();
    // Crop CSS rounded corners: their subpixel antialiasing can change after modal scroll locking.
    return page.screenshot({path: testInfo.outputPath(`${name}.png`), clip: {
      x: box.x + 16, y: box.y + 16, width: box.width - 32, height: box.height - 32,
    }});
  };
  const beforePan = await screenshotPlot("before-pan");
  const bounds = await region.locator("canvas").boundingBox();
  await page.mouse.move(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
  await page.mouse.down();
  await page.mouse.move(bounds.x + bounds.width / 2 + 35, bounds.y + bounds.height / 2 + 20,
    {steps: 8});
  await page.mouse.up();
  await page.mouse.move(0, 0);
  await expect(region.locator("[data-cell-type-tooltip]")).toBeHidden();
  const cameraImage = await screenshotPlot("camera-before");
  expect(cameraImage.equals(beforePan)).toBe(false);
  await glue.click();
  await expect(region).toHaveAttribute("data-visualization-state", "ready");
  await expect(glue).toHaveAttribute("aria-pressed", "true");
  await expect(rna).toHaveAttribute("aria-pressed", "false");
  expect(await canvas.evaluate((element) => element.isConnected)).toBe(true);
  await expect(region.locator("[data-cell-type-sample]")).toHaveValue(sample);
  await expect(domains.first().locator("input")).toBeChecked();
  await domains.first().locator("input").uncheck();
  const excluded = region.locator(".cell-type-legend-item").filter({hasText: "Not analyzed"});
  if (await excluded.count()) await expect(excluded.locator("input")).not.toBeChecked();
  await region.locator("[data-visualization-method]").click();
  const modal = page.locator(await region.locator("[data-visualization-method]").getAttribute("data-bs-target"));
  await expect(modal).toBeVisible();
  await expect(modal).toContainText("SpatialGlue");
  await expect(modal).toContainText("igraph");
  await expect(modal).toContainText("rna");
  if (process.env.ISCDC_GLUE_UNUSED_MODALITY) {
    await expect(modal).toContainText("Unused modalities");
    await expect(modal.getByText(process.env.ISCDC_GLUE_UNUSED_MODALITY, {exact: true})).toBeVisible();
  }
  await modal.getByRole("button", {name: "Close", exact: true}).click();
  await expect(modal).not.toBeVisible();
  await rna.click();
  await expect(region).toHaveAttribute("data-visualization-state", "ready");
  await expect(domains.first().locator("input")).not.toBeChecked();
  const restoredImage = await screenshotPlot("camera-after");
  await testInfo.attach("camera-before", {body: cameraImage, contentType: "image/png"});
  await testInfo.attach("camera-after", {body: restoredImage, contentType: "image/png"});
  expect(restoredImage.equals(cameraImage)).toBe(true);
  await glue.click();
  await expect(region).toHaveAttribute("data-visualization-state", "ready");
  await expect(domains.first().locator("input")).not.toBeChecked();
  expect(errors).toEqual([]);
});


test("four SpatialGLUE combinations preserve camera and independent legends", async ({page}, testInfo) => {
  test.skip(!databasePath || !process.env.ISCDC_GLUE_FOUR_COMBINATIONS, "Requires a four-combination fixture");
  test.setTimeout(120000);
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/favicon.ico", (route) => route.fulfill({status: 204}));
  await page.goto(databasePath);
  const region = page.locator("#cell-type-visualization");
  await region.scrollIntoViewIfNeeded();
  await region.getByRole("button", {name: "SpatialGLUE domains", exact: true}).click();
  await expect(region).toHaveAttribute("data-visualization-state", "ready", {timeout: 90000});
  const select = region.locator("[data-spatialglue-combination]");
  await expect(select.locator("option")).toHaveCount(4);
  const values = await select.locator("option").evaluateAll((options) => options.map((o) => o.value));
  const first = values[0];
  const checkbox = region.locator(".cell-type-legend-item input").filter({visible: true});
  const domain = region.locator(".cell-type-legend-item").filter({hasText: /^Domain /}).first().locator("input");
  await domain.uncheck();
  const canvas = await region.locator("canvas").elementHandle();
  const sample = await region.locator("[data-cell-type-sample]").inputValue();
  for (const value of values.slice(1)) {
    await select.selectOption(value);
    await expect(region).toHaveAttribute("data-visualization-state", "ready");
    await expect(domain).toBeChecked();
    await domain.uncheck();
    await expect(region.locator("[data-cell-type-sample]")).toHaveValue(sample);
    expect(await canvas.evaluate((element) => element.isConnected)).toBe(true);
  }
  await select.selectOption(first);
  await expect(region).toHaveAttribute("data-visualization-state", "ready");
  await expect(domain).not.toBeChecked();
  for (const value of values) {
    await select.selectOption(value);
    await expect(region).toHaveAttribute("data-visualization-state", "ready");
    await expect(domain).not.toBeChecked();
  }
  expect(await checkbox.count()).toBeGreaterThan(0);
  expect(errors).toEqual([]);
  await page.screenshot({path: testInfo.outputPath("four-combinations-desktop.png"), fullPage: true});
  await page.setViewportSize({width: 390, height: 844});
  await region.scrollIntoViewIfNeeded();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({path: testInfo.outputPath("four-combinations-mobile.png"), fullPage: true});
});
