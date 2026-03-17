import "./styles.css";
import vegaEmbed from "vega-embed";
import { Warn } from "vega";
import { WasmContextManager } from "./context";
import { EditorManager } from "./editor";
import { TableManager } from "./tableManager";
import { examples } from "./examples";

// State
const contextManager = new WasmContextManager();
const editorManager = new EditorManager();
let tableManager: TableManager;

// DOM elements
const statusEl = document.getElementById("status")!;
const editorContainer = document.getElementById("editor-container")!;
const vizOutput = document.getElementById("viz-output")!;
const errorMessages = document.getElementById("error-messages")!;
const tableList = document.getElementById("table-list")!;
const csvUpload = document.getElementById("csv-upload") as HTMLInputElement;
const examplesList = document.getElementById("examples-list")!;

function setStatus(message: string, type: "loading" | "success" | "error") {
  statusEl.textContent = message;
  statusEl.className = type;
}

function showProblems(
  errors: string[],
  warnings: string[],
) {
  errorMessages.innerHTML = "";
  for (const msg of errors) {
    const div = document.createElement("div");
    div.className = "error-message";
    div.textContent = msg;
    errorMessages.appendChild(div);
  }
  for (const msg of warnings) {
    const div = document.createElement("div");
    div.className = "warning-message";
    div.textContent = msg;
    errorMessages.appendChild(div);
  }
}

interface SqlResult {
  columns: string[];
  rows: string[][];
  total_rows: number;
  truncated: boolean;
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderTable(data: SqlResult): string {
  const ths = data.columns.map((c) => `<th>${escapeHtml(c)}</th>`).join("");
  const bodyRows = data.rows
    .map(
      (row) =>
        `<tr>${row.map((v) => `<td>${escapeHtml(v)}</td>`).join("")}</tr>`,
    )
    .join("");
  const truncationRow = data.truncated
    ? `<tr class="truncation-row"><td colspan="${data.columns.length}">Showing ${data.rows.length} of ${data.total_rows} rows</td></tr>`
    : "";
  return `<table class="ggsql-table"><thead><tr>${ths}</tr></thead><tbody>${bodyRows}${truncationRow}</tbody></table>`;
}

async function executeQuery(query: string) {
  if (!query.trim()) {
    showProblems([], []);
    vizOutput.innerHTML =
      '<p style="color: #005F73; text-align: center; padding: 40px;">Enter a query to visualize</p>';
    return;
  }

  try {
    setStatus("Executing query...", "loading");

    if (contextManager.hasVisual(query)) {
      const result = contextManager.execute(query);
      const spec = JSON.parse(result);

      vizOutput.innerHTML = "";

      const warnings: string[] = [];
      let _level = Warn;
      const logger = {
        level(_: number) { if (arguments.length) { _level = _; return this; } return _level; },
        error: (...args: any[]) => { console.error(...args); return logger; },
        warn: (...args: any[]) => { warnings.push(args.map(String).join(" ")); return logger; },
        info: () => logger,
        debug: () => logger,
      };

      await vegaEmbed(vizOutput, spec, {
        actions: {
          export: true,
          source: false,
          compiled: false,
          editor: false,
        },
        renderer: "svg",
        logger: logger as any,
      });

      showProblems([], warnings);
    } else {
      const result = JSON.parse(contextManager.executeSql(query));
      vizOutput.innerHTML = renderTable(result);
      showProblems([], []);
    }

    setStatus("Query executed successfully", "success");
  } catch (error: any) {
    console.error("Query execution error:", error);
    showProblems([error.toString()], []);
    setStatus("Query error", "error");
  }
}

// File upload handlers
csvUpload.addEventListener("change", async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;

  try {
    setStatus("Uploading data...", "loading");
    await tableManager.uploadFile(file);
    setStatus("Uploaded: " + file.name, "success");
    csvUpload.value = "";
  } catch (error: any) {
    showProblems(["Upload failed: " + error], []);
    setStatus("Upload error", "error");
  }
});

function initializeExamples() {
  let currentSection = "";
  examples.forEach((example) => {
    if (example.section !== currentSection) {
      currentSection = example.section;
      const header = document.createElement("div");
      header.className = "example-section-header";
      header.textContent = currentSection;
      examplesList.appendChild(header);
    }
    const button = document.createElement("button");
    button.className = "example-button";
    button.textContent = example.name;
    button.onclick = () => {
      editorManager.setValue(example.query);
      //executeQuery(example.query);
    };
    examplesList.appendChild(button);
  });
}

function initializeMobileExamples() {
  const select = document.getElementById(
    "mobile-example-select",
  ) as HTMLSelectElement;

  let currentSection = "";
  let optgroup: HTMLOptGroupElement | null = null;
  examples.forEach((example, index) => {
    if (example.section !== currentSection) {
      currentSection = example.section;
      optgroup = document.createElement("optgroup");
      optgroup.label = currentSection;
      select.appendChild(optgroup);
    }
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = example.name;
    optgroup!.appendChild(option);
  });

  select.addEventListener("change", () => {
    const idx = parseInt(select.value, 10);
    if (!isNaN(idx) && examples[idx]) {
      editorManager.setValue(examples[idx].query);
    }
  });
}

async function main() {
  try {
    setStatus("Loading WASM module...", "loading");
    await contextManager.initialize();

    // Load builtin datasets
    setStatus("Loading builtin datasets...", "loading");
    await contextManager.registerBuiltinDatasets();

    setStatus("Initializing editor...", "loading");
    await editorManager.initialize(editorContainer, examples[0].query);

    tableManager = new TableManager(tableList, contextManager);
    tableManager.onClickTable((name) => {
      editorManager.setValue(`SELECT * FROM ${name}`);
    });
    tableManager.refresh();

    initializeExamples();
    initializeMobileExamples();

    editorManager.onChange((query) => {
      executeQuery(query);
    });

    setStatus("Ready", "success");

    executeQuery(examples[0].query);
  } catch (error: any) {
    console.error("Initialization error:", error);
    setStatus("Initialization failed", "error");
    showProblems(["Failed to initialize: " + error], []);
  }
}

main();
