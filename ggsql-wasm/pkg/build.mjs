import * as esbuild from "esbuild";
import { copyFileSync, mkdirSync, readdirSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const isWatch = process.argv.includes("--watch");
const distDir = join(__dirname, "dist");
const fontsDir = join(distDir, "fonts");

mkdirSync(fontsDir, { recursive: true });
for (const file of readdirSync(join(__dirname, "../fonts"))) {
  copyFileSync(join(__dirname, "../fonts", file), join(fontsDir, file));
}

const buildOptions = {
  entryPoints: [join(__dirname, "src/ggsql.ts")],
  bundle: true,
  outfile: join(distDir, "ggsql.js"),
  format: "esm",
  platform: "browser",
  target: "es2022",
  sourcemap: true,
  external: ["./ggsql_wasm.js"],
};

if (isWatch) {
  console.log("Starting package watch mode...");
  const ctx = await esbuild.context(buildOptions);
  await ctx.watch();
  console.log("Watching for changes...");
} else {
  console.log("Building package client...");
  await esbuild.build(buildOptions);
  console.log("Build complete!");
}
