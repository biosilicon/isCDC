import {expect, test} from "@playwright/test";

test("database detail cell type visualization loads and pans without browser errors", async ({page}) => {
  const errors = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));

  await page.goto(process.env.ISCDC_DATABASE_PATH || "/");
  const region = page.locator("#cell-type-visualization");
  if (await region.count() === 0) {
    test.skip(true, "The selected page has no published cell type visualization");
  }
  await region.evaluate((element) => element.scrollIntoView({block: "center"}));
  const canvas = region.locator("canvas");
  await expect.poll(async () => {
    if (await region.getAttribute("hidden") !== null) return "hidden";
    return (await canvas.isVisible()) ? "ready" : "loading";
  }, {timeout: 15_000}).not.toBe("loading");
  if (await region.getAttribute("hidden") !== null) {
    expect(errors).toEqual([]);
    return;
  }
  await expect(region.locator("[data-cell-type-reset]")).toBeVisible();
  await expect(region.locator("[data-cell-type-legend]")).toBeVisible();
  await expect(canvas).toBeVisible({timeout: 15_000});
  const box = await canvas.boundingBox();
  expect(box).not.toBeNull();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 30, box.y + box.height / 2 + 20);
  await page.mouse.up();
  expect(errors).toEqual([]);
});
