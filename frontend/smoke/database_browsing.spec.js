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
  const entryId = await entryLink.textContent();
  await page.screenshot({
    path: resolve(screenshotDirectory, "databases-entry-mobile.png"),
    fullPage: true,
  });

  await entryLink.click();
  await expect(page.getByRole("heading", {name: entryId.trim(), exact: true})).toBeVisible();
  await expect(page.getByRole("heading", {name: "Datasets in this entry"})).toBeVisible();
  await page.screenshot({
    path: resolve(screenshotDirectory, "database-entry-detail-mobile.png"),
    fullPage: true,
  });
});
