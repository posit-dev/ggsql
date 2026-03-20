import * as monaco from "monaco-editor";
import {
  createOnigScanner,
  createOnigString,
  loadWASM,
} from "vscode-oniguruma";
import { Registry, parseRawGrammar, type IGrammar } from "vscode-textmate";
import { WASM_BASE } from "../wasmBase";

// Must be set before any Monaco editor is created
(self as any).MonacoEnvironment = {
  getWorkerUrl: (_moduleId: string, _label: string) =>
    WASM_BASE + "editor.worker.js",
};

// Map TextMate scope names to Monaco theme token colors
const SCOPE_TO_TOKEN: [string, string][] = [
  ["comment", "comment"],
  ["string", "string"],
  ["constant.numeric", "number"],
  ["constant.language", "keyword"],
  ["keyword", "keyword"],
  ["support.function", "type"],
  ["support.type.geom", "type"],
  ["support.type.aesthetic", "variable"],
  ["support.type.coord", "type"],
  ["support.type.theme", "type"],
  ["support.type.property", "variable"],
  ["constant.language.scale-type", "type"],
  ["keyword.operator", "operator"],
  ["punctuation", "delimiter"],
];

function scopeToMonacoToken(scopes: string[]): string {
  for (let i = scopes.length - 1; i >= 0; i--) {
    const scope = scopes[i];
    for (const [pattern, token] of SCOPE_TO_TOKEN) {
      if (scope.startsWith(pattern)) {
        return token;
      }
    }
  }
  return "";
}

// Singleton grammar initialization
let grammarPromise: Promise<IGrammar | null> | null = null;

async function initTextMateGrammar(): Promise<IGrammar | null> {
  const onigWasm = await fetch(WASM_BASE + "onig.wasm");
  const onigBuffer = await onigWasm.arrayBuffer();
  await loadWASM(onigBuffer);

  const registry = new Registry({
    onigLib: Promise.resolve({
      createOnigScanner,
      createOnigString,
    }),
    loadGrammar: async (scopeName: string) => {
      if (scopeName === "source.ggsql") {
        const response = await fetch(WASM_BASE + "ggsql.tmLanguage.json");
        const grammarText = await response.text();
        return parseRawGrammar(grammarText, "ggsql.tmLanguage.json");
      }
      return null;
    },
  });

  return registry.loadGrammar("source.ggsql");
}

function getGrammar(): Promise<IGrammar | null> {
  if (!grammarPromise) {
    grammarPromise = initTextMateGrammar();
  }
  return grammarPromise;
}

// Define Pygments-style theme for Monaco
monaco.editor.defineTheme("ggsql-pygments", {
  base: "vs",
  inherit: true,
  rules: [
    { token: "comment", foreground: "408080", fontStyle: "italic" },
    { token: "string", foreground: "BA2121" },
    { token: "number", foreground: "666666" },
    { token: "keyword", foreground: "008000", fontStyle: "bold" },
    { token: "type", foreground: "008000" },
    { token: "variable", foreground: "19177C" },
    { token: "operator", foreground: "666666" },
    { token: "delimiter", foreground: "000000" },
  ],
  colors: {
    "editor.background": "#f7f7f7",
  },
});

let languageRegistered = false;

async function ensureLanguageRegistered(): Promise<void> {
  if (languageRegistered) return;
  languageRegistered = true;

  monaco.languages.register({ id: "ggsql" });

  monaco.languages.setLanguageConfiguration("ggsql", {
    comments: {
      lineComment: "--",
      blockComment: ["/*", "*/"],
    },
    brackets: [
      ["{", "}"],
      ["[", "]"],
      ["(", ")"],
    ],
    autoClosingPairs: [
      { open: "{", close: "}" },
      { open: "[", close: "]" },
      { open: "(", close: ")" },
      { open: "'", close: "'", notIn: ["string", "comment"] },
      { open: '"', close: '"', notIn: ["string", "comment"] },
    ],
    surroundingPairs: [
      { open: "{", close: "}" },
      { open: "[", close: "]" },
      { open: "(", close: ")" },
      { open: "'", close: "'" },
      { open: '"', close: '"' },
    ],
  });

  const grammar = await getGrammar();
  if (grammar) {
    monaco.languages.setTokensProvider("ggsql", {
      getInitialState: () => new TMState(null),
      tokenize: (
        line: string,
        state: TMState
      ): monaco.languages.ILineTokens => {
        const result = grammar.tokenizeLine(line, state.ruleStack);
        const tokens: monaco.languages.IToken[] = result.tokens.map((t) => ({
          startIndex: t.startIndex,
          scopes: scopeToMonacoToken(t.scopes),
        }));
        return {
          tokens,
          endState: new TMState(result.ruleStack),
        };
      },
    });
  }
}

// TextMate state wrapper for Monaco
class TMState implements monaco.languages.IState {
  constructor(public ruleStack: any) {}

  clone(): TMState {
    return new TMState(this.ruleStack);
  }

  equals(other: monaco.languages.IState): boolean {
    if (!(other instanceof TMState)) return false;
    if (!this.ruleStack && !other.ruleStack) return true;
    if (!this.ruleStack || !other.ruleStack) return false;
    return this.ruleStack.equals(other.ruleStack);
  }
}

export interface EditorInstance {
  getValue(): string;
  setValue(value: string): void;
  editor: monaco.editor.IStandaloneCodeEditor;
}

const LINE_HEIGHT = 20;
const PADDING_TOP = 8;
const PADDING_BOTTOM = 8;
const MAX_EDITOR_HEIGHT = 400;

function editorHeight(lineCount: number): number {
  const contentHeight = lineCount * LINE_HEIGHT + PADDING_TOP + PADDING_BOTTOM;
  return Math.min(contentHeight, MAX_EDITOR_HEIGHT);
}

export async function createEditor(
  container: HTMLElement,
  initialValue: string
): Promise<EditorInstance> {
  await ensureLanguageRegistered();

  const lineCount = initialValue.split("\n").length;
  container.style.height = editorHeight(lineCount) + "px";

  const editor = monaco.editor.create(container, {
    value: initialValue,
    language: "ggsql",
    theme: "ggsql-pygments",
    automaticLayout: true,
    minimap: { enabled: false },
    fontSize: 13,
    lineNumbers: "on",
    glyphMargin: false,
    folding: false,
    lineNumbersMinChars: 3,
    scrollBeyondLastLine: false,
    wordWrap: "on",
    padding: { top: PADDING_TOP, bottom: PADDING_BOTTOM },
    renderLineHighlightOnlyWhenFocus: true,
    overviewRulerLanes: 0,
    hideCursorInOverviewRuler: true,
    overviewRulerBorder: false,
    scrollbar: {
      vertical: "hidden",
      horizontal: "hidden",
      alwaysConsumeMouseWheel: false,
    },
  });

  // Auto-resize editor height to content
  editor.onDidContentSizeChange(() => {
    const newLineCount = editor.getModel()?.getLineCount() || lineCount;
    container.style.height = editorHeight(newLineCount) + "px";
    editor.layout();
  });

  return {
    getValue: () => editor.getValue(),
    setValue: (value: string) => editor.setValue(value),
    editor,
  };
}
