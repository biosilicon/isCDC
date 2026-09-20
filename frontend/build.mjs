import {build} from "esbuild";
import {readFile, writeFile} from "node:fs/promises";
import {fileURLToPath} from "node:url";

const output = fileURLToPath(new URL("../assets/static/cell_type_visualization.js", import.meta.url));
await build({
  entryPoints: [fileURLToPath(new URL("src/cell_type_visualization.js", import.meta.url))],
  bundle: true, minify: true, treeShaking: true, platform: "browser", target: "es2022", outfile: output,
});
// Upstream GLSL template strings contain trailing whitespace; keep committed bundles clean.
const source = await readFile(output, "utf8");
await writeFile(output, source.replace(/[\t ]+$/gm, ""));
