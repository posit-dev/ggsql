import "./styles.css";
import vegaEmbed from "vega-embed";
import { WasmContextManager } from "../context";
import { createEditor, type EditorInstance } from "./editor";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface CellInfo {
  query: string;
  rewrittenQuery: string;
  cellDiv: HTMLElement;
  codeScaffold: HTMLElement;
  visId: string | null;
  visContainer: HTMLElement | null;
  result: string | null;
  succeeded: boolean;
  error: string | null;
  editor: EditorInstance | null;
  errorDisplay: HTMLElement | null;
  errorTimerId: number | null;
}

// ---------------------------------------------------------------------------
// Site root
// ---------------------------------------------------------------------------

const SITE_ROOT = (() => {
  const meta = document.querySelector('meta[name="quarto:offset"]');
  return meta?.getAttribute("content") || "./";
})();

// ---------------------------------------------------------------------------
// CSV rewriting
// ---------------------------------------------------------------------------

function findCsvReferences(queries: string[]): string[] {
  const csvFiles = new Set<string>();
  const re = /(?:FROM|JOIN)\s+'([^']+\.csv)'/gi;
  for (const q of queries) {
    let m: RegExpExecArray | null;
    while ((m = re.exec(q)) !== null) {
      csvFiles.add(m[1]);
    }
  }
  return Array.from(csvFiles);
}

function csvTableName(filename: string): string {
  return filename.replace(/\.csv$/i, "");
}

function rewriteCsvRefs(query: string): string {
  return query.replace(
    /(?<=FROM|JOIN)\s+'([^']+)\.csv'/gi,
    (_match, name) => ` ${name}`
  );
}

// ---------------------------------------------------------------------------
// Vega embed options
// ---------------------------------------------------------------------------

const VEGA_EMBED_OPTS = {
  actions: { export: true, source: false, compiled: false, editor: false },
  renderer: "svg" as const,
};

// ---------------------------------------------------------------------------
// Phase 1: Gather cell metadata from the DOM (no mutations)
// ---------------------------------------------------------------------------

function gatherCells(): CellInfo[] {
  const cells: CellInfo[] = [];

  const codeEls = document.querySelectorAll<HTMLElement>(
    "div.sourceCode.cell-code code.sourceCode.ggsql"
  );

  for (const codeEl of codeEls) {
    const query = codeEl.textContent?.trim() || "";
    if (!query) continue;

    const cellDiv = codeEl.closest<HTMLElement>(".cell");
    if (!cellDiv) continue;

    const codeScaffold =
      cellDiv.querySelector<HTMLElement>(".code-copy-outer-scaffold") ||
      cellDiv.querySelector<HTMLElement>(".sourceCode.cell-code");
    if (!codeScaffold) continue;

    const outputDiv = cellDiv.querySelector<HTMLElement>(
      ".cell-output.cell-output-display"
    );
    let visId: string | null = null;
    let visContainer: HTMLElement | null = null;

    if (outputDiv) {
      const visCandidates = outputDiv.querySelectorAll<HTMLElement>(
        'div[id^="vis-"]'
      );
      if (visCandidates.length > 0) {
        visContainer = visCandidates[0];
        visId = visContainer.id;
      }
    }

    cells.push({
      query,
      rewrittenQuery: rewriteCsvRefs(query),
      cellDiv,
      codeScaffold,
      visId,
      visContainer,
      result: null,
      succeeded: false,
      error: null,
      editor: null,
      errorDisplay: null,
      errorTimerId: null,
    });
  }

  return cells;
}

// ---------------------------------------------------------------------------
// Phase 2: Initialize WASM context and execute all cells
// ---------------------------------------------------------------------------

async function initAndExecute(
  cells: CellInfo[]
): Promise<WasmContextManager | null> {
  const ctx = new WasmContextManager();

  console.log("[ggsql-quarto] Loading WebAssembly…");
  try {
    await ctx.initialize();
  } catch (e) {
    console.error("[ggsql-quarto] WASM init failed:", e);
    return null;
  }

  console.log("[ggsql-quarto] Registering datasets…");
  try {
    await ctx.registerBuiltinDatasets();
  } catch (e) {
    console.error("[ggsql-quarto] Builtin dataset registration failed:", e);
    return null;
  }

  const csvFiles = findCsvReferences(cells.map((c) => c.query));
  if (csvFiles.length > 0) {
    console.log("[ggsql-quarto] Loading data files:", csvFiles.join(", "));
    for (const file of csvFiles) {
      try {
        const resp = await fetch(SITE_ROOT + file);
        if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${file}`);
        const bytes = new Uint8Array(await resp.arrayBuffer());
        ctx.registerCSV(csvTableName(file), bytes);
      } catch (e) {
        console.error(`[ggsql-quarto] Failed to load CSV '${file}':`, e);
        return null;
      }
    }
  }

  const total = cells.length;
  console.log(`[ggsql-quarto] Executing ${total} cells…`);
  for (let i = 0; i < total; i++) {
    const cell = cells[i];
    try {
      if (ctx.hasVisual(cell.rewrittenQuery)) {
        cell.result = ctx.execute(cell.rewrittenQuery);
      } else {
        ctx.executeSql(cell.rewrittenQuery);
        cell.result = null;
      }
      cell.succeeded = true;
    } catch (e: any) {
      cell.succeeded = false;
      cell.error = String(e);
      console.warn(
        `[ggsql-quarto] Cell ${i + 1}/${total} failed:`,
        cell.query.slice(0, 80),
        e
      );
    }
  }

  const succeeded = cells.filter((c) => c.succeeded).length;
  console.log(`[ggsql-quarto] ${succeeded}/${total} cells succeeded`);

  return ctx;
}

// ---------------------------------------------------------------------------
// Phase 3: Mutate DOM — replace succeeded cells with editors, render results
// ---------------------------------------------------------------------------

const DEBOUNCE_MS = 100;

async function applyEditors(
  cells: CellInfo[],
  ctx: WasmContextManager
): Promise<void> {
  for (const cell of cells) {
    if (!cell.succeeded) continue;

    const wrapper = document.createElement("div");
    wrapper.className = "ggsql-editor-wrapper";

    const editorContainer = document.createElement("div");
    editorContainer.className = "ggsql-editor-container";
    wrapper.appendChild(editorContainer);

    const errorDisplay = document.createElement("div");
    errorDisplay.className = "ggsql-error-display";
    wrapper.appendChild(errorDisplay);
    cell.errorDisplay = errorDisplay;

    wrapper.appendChild(errorDisplay);

    cell.codeScaffold.replaceWith(wrapper);

    const editorInst = await createEditor(editorContainer, cell.query, SITE_ROOT);
    cell.editor = editorInst;

    if (cell.result && cell.visId && cell.visContainer) {
      try {
        const spec = JSON.parse(cell.result);
        cell.visContainer.innerHTML = "";
        await vegaEmbed("#" + cell.visId, spec, VEGA_EMBED_OPTS);
      } catch (e) {
        console.warn("[ggsql-quarto] vegaEmbed failed for", cell.visId, e);
      }
    }

    // Re-execute on every edit, debounced
    let debounceTimer: number | undefined;
    editorInst.editor.onDidChangeModelContent(() => {
      clearTimeout(debounceTimer);
      debounceTimer = window.setTimeout(() => {
        executeCell(cell, editorInst, ctx);
      }, DEBOUNCE_MS);
    });
  }
}

const ERROR_DELAY_MS = 1000;

function clearError(cell: CellInfo): void {
  if (cell.errorTimerId !== null) {
    clearTimeout(cell.errorTimerId);
    cell.errorTimerId = null;
  }
  const el = cell.errorDisplay!;
  el.innerHTML = "";
  el.classList.remove("visible", "collapsed");
}

function showError(cell: CellInfo, message: string): void {
  clearError(cell);
  cell.errorTimerId = window.setTimeout(() => {
    cell.errorTimerId = null;
    const el = cell.errorDisplay!;
    el.innerHTML = "";

    const toggle = document.createElement("span");
    toggle.className = "ggsql-error-toggle";
    toggle.textContent = "\u25B6";

    const text = document.createElement("span");
    text.className = "ggsql-error-text";
    text.textContent = message;

    el.appendChild(toggle);
    el.appendChild(text);
    el.classList.add("visible", "collapsed");

    el.onclick = () => {
      const isCollapsed = el.classList.toggle("collapsed");
      toggle.textContent = isCollapsed ? "\u25B6" : "\u25BC";
    };
  }, ERROR_DELAY_MS);
}

async function executeCell(
  cell: CellInfo,
  editorInst: EditorInstance,
  ctx: WasmContextManager
): Promise<void> {
  clearError(cell);

  const currentQuery = rewriteCsvRefs(editorInst.getValue());

  try {
    if (ctx.hasVisual(currentQuery)) {
      const result = ctx.execute(currentQuery);
      const spec = JSON.parse(result);

      if (cell.visContainer && cell.visId) {
        cell.visContainer.innerHTML = "";
        await vegaEmbed("#" + cell.visId, spec, VEGA_EMBED_OPTS);
      }
    } else {
      ctx.executeSql(currentQuery);
    }
  } catch (e: any) {
    showError(cell, String(e));
  }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

async function main() {
  const cells = gatherCells();
  if (cells.length === 0) return;

  console.log(`[ggsql-quarto] Found ${cells.length} ggsql cells`);

  const ctx = await initAndExecute(cells);
  if (!ctx) return;

  const anySucceeded = cells.some((c) => c.succeeded);
  if (!anySucceeded) return;

  await applyEditors(cells, ctx);
  console.log("[ggsql-quarto] Done");
}

main().catch((e) => {
  console.error("[ggsql-quarto] Unexpected error:", e);
});
