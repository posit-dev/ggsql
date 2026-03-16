import * as esbuild from "esbuild";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const isWatch = process.argv.includes("--watch");

const buildOptions = {
  entryPoints: [join(__dirname, "src/index.ts")],
  bundle: true,
  outfile: join(__dirname, "dist/lib.js"),
  format: "esm",
  platform: "browser",
  target: "es2020",
  sourcemap: true,
};

if (isWatch) {
  console.log("Starting watch mode...");
  const ctx = await esbuild.context(buildOptions);
  await ctx.watch();
  console.log("Watching for changes...");
} else {
  console.log("Building library...");
  await esbuild.build(buildOptions);
  console.log("Build complete!");
}
