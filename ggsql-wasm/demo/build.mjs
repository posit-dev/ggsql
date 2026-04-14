import * as esbuild from "esbuild";
import { copyFileSync, mkdirSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const isWatch = process.argv.includes("--watch");
const distDir = join(__dirname, "dist");

// Ensure dist/ directory exists
mkdirSync(distDir, { recursive: true });

// Copy static files
console.log("Copying static files...");
copyFileSync(join(__dirname, "src/index.qmd"), join(distDir, "index.qmd"));
copyFileSync(
  join(__dirname, "../pkg/ggsql_wasm_bg.wasm"),
  join(distDir, "ggsql_wasm_bg.wasm"),
);
copyFileSync(
  join(__dirname, "node_modules/vscode-oniguruma/release/onig.wasm"),
  join(distDir, "onig.wasm"),
);
copyFileSync(
  join(__dirname, "../../ggsql-vscode/syntaxes/ggsql.tmLanguage.json"),
  join(distDir, "ggsql.tmLanguage.json"),
);

// Build Monaco editor web worker
console.log("Building Monaco editor worker...");
await esbuild.build({
  entryPoints: [
    join(
      __dirname,
      "node_modules/monaco-editor/esm/vs/editor/editor.worker.js",
    ),
  ],
  bundle: true,
  outfile: join(distDir, "editor.worker.js"),
  format: "iife",
});

// Shared build options
const sharedOptions = {
  bundle: true,
  format: "esm",
  platform: "browser",
  target: "es2020",
  sourcemap: true,
  nodePaths: [join(__dirname, "node_modules")],
  loader: {
    ".ttf": "file",
  },
};

// Build playground bundle
const playgroundOptions = {
  ...sharedOptions,
  entryPoints: [join(__dirname, "src/main.ts")],
  outfile: join(distDir, "bundle.js"),
};

// Build quarto integration bundle
const quartoOptions = {
  ...sharedOptions,
  entryPoints: [join(__dirname, "src/quarto/main.ts")],
  outfile: join(distDir, "quarto.js"),
  loader: {
    ...sharedOptions.loader,
    ".css": "css",
  },
};

if (isWatch) {
  console.log("Starting watch mode...");
  const playgroundCtx = await esbuild.context(playgroundOptions);
  const quartoCtx = await esbuild.context(quartoOptions);
  await Promise.all([playgroundCtx.watch(), quartoCtx.watch()]);
  console.log("Watching for changes...");
} else {
  console.log("Building bundles...");
  await Promise.all([
    esbuild.build(playgroundOptions),
    esbuild.build(quartoOptions),
  ]);
  console.log("Build complete!");
}
