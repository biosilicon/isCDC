import {mkdir} from "node:fs/promises";
import {resolve} from "node:path";

import {expect, test} from "@playwright/test";

const screenshotDirectory = resolve("../temp/browser_qa");

test.beforeAll(async () => {
  await mkdir(screenshotDirectory, {recursive: true});
});

test("database catalogue switches between Entry and Dataset browsing", async ({page}) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));

  await page.goto("/databases");
  await expect(page.getByRole("link", {name: "By entry"})).toHaveAttribute(
    "aria-current",
    "page",
  );
  await expect(page.getByText(/matching entr(?:y|ies)/)).toBeVisible();
  await page.screenshot({
    path: resolve(screenshotDirectory, "databases-entry-desktop.png"),
    fullPage: true,
  });

  await page.getByRole("link", {name: "By dataset"}).click();
  await expect(page).toHaveURL(/view=datasets/);
  await expect(page.getByText(/matching datasets?/)).toBeVisible();
  await page.screenshot({
    path: resolve(screenshotDirectory, "databases-dataset-desktop.png"),
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("Entry list and detail remain usable on a narrow viewport", async ({page}) => {
  await page.setViewportSize({width: 375, height: 812});
  await page.goto("/databases");
  const entryLink = page.locator(".database-entry-card h2 a").first();
  await expect(entryLink).toBeVisible();
  const entryName = await entryLink.textContent();
  await page.screenshot({
    path: resolve(screenshotDirectory, "databases-entry-mobile.png"),
    fullPage: true,
  });

  await entryLink.click();
  await expect(page.getByRole("heading", {name: entryName.trim(), exact: true})).toBeVisible();
  await expect(page.getByRole("heading", {name: "Datasets in this entry"})).toBeVisible();
  await page.screenshot({
    path: resolve(screenshotDirectory, "database-entry-detail-mobile.png"),
    fullPage: true,
  });
});

for (const [viewportName, width] of [["desktop", 1440], ["mobile", 375]]) {
  test(`Entry names appear in search, detail and breadcrumbs on ${viewportName}`, async ({page}) => {
    test.setTimeout(90_000);
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") errors.push(message.text());
    });
    await page.setViewportSize({width, height: 900});
    const name = "Xenium Human Head and Neck Tumors — RNA and TCR";
    await page.goto(`/databases?q=${encodeURIComponent(name)}`);
    await expect(page.getByText("1 matching entry", {exact: true})).toBeVisible();
    const link = page.getByRole("link", {name, exact: true});
    await expect(link).toHaveAttribute("href", /\/databases\/entries\/S044$/);
    await expect(page.getByText("Entry ID: S044", {exact: true})).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth))
      .toBe(true);
    await page.screenshot({
      path: resolve(screenshotDirectory, `entry-names-list-${viewportName}.png`), fullPage: true,
    });
    await link.click();
    await expect(page.getByRole("heading", {name, exact: true})).toBeVisible();
    await expect(page).toHaveTitle(`${name} · isCDC`);
    await expect(page.getByText("Entry ID: S044", {exact: true})).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth))
      .toBe(true);
    await page.screenshot({
      path: resolve(screenshotDirectory, `entry-names-detail-${viewportName}.png`),
    });
    const api = await (await page.request.get("/api/database-entries/S044")).json();
    expect(api.display_name).toBe(name);
    expect(api.slide_count).toBe(17);
    await page.goto(`/databases/${api.datasets[0].dataset_id}`);
    await expect(page.locator(".database-breadcrumb").getByRole("link", {name, exact: true}))
      .toBeVisible();
    await page.screenshot({
      path: resolve(screenshotDirectory, `entry-names-breadcrumb-${viewportName}.png`),
    });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth))
      .toBe(true);
    expect(errors).toEqual([]);
  });
}
