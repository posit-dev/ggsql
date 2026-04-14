import * as monaco from "monaco-editor";
import { createOnigScanner, createOnigString, loadWASM } from "vscode-oniguruma";
import { Registry, parseRawGrammar, type IGrammar } from "vscode-textmate";
import { WASM_BASE } from "./wasmBase";

// Must be set before any Monaco editor is created
(self as any).MonacoEnvironment = {
  getWorkerUrl: (_moduleId: string, _label: string) => WASM_BASE + "editor.worker.js",
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
  // Walk scopes from most specific to least
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
  colors: {},
});

async function initTextMateGrammar(): Promise<IGrammar | null> {
  // Load oniguruma WASM
  const onigWasm = await fetch(WASM_BASE + "onig.wasm");
  const onigBuffer = await onigWasm.arrayBuffer();
  await loadWASM(onigBuffer);

  // Create the TextMate registry
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

export class EditorManager {
  private editor: monaco.editor.IStandaloneCodeEditor | null = null;
  private onChangeCallback: ((query: string) => void) | null = null;
  private changeTimeoutId: number | null = null;

  async initialize(
    container: HTMLElement,
    initialValue: string,
  ): Promise<void> {
    // Register ggsql language
    monaco.languages.register({ id: "ggsql" });

    // Apply language configuration from ggsql-vscode/language-configuration.json
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

    // Load TextMate grammar for tokenization
    const grammar = await initTextMateGrammar();
    if (grammar) {
      // Create a custom tokens provider using the TextMate grammar
      monaco.languages.setTokensProvider("ggsql", {
        getInitialState: () => new TMState(null),
        tokenize: (line: string, state: TMState): monaco.languages.ILineTokens => {
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

    // Create editor
    this.editor = monaco.editor.create(container, {
      value: initialValue,
      language: "ggsql",
      theme: "ggsql-pygments",
      automaticLayout: true,
      minimap: { enabled: false },
      fontSize: 14,
      lineNumbers: "on",
      scrollBeyondLastLine: false,
      wordWrap: "on",
      padding: { top: 10 },
    });

    // Set up change listener with debounce
    this.editor.onDidChangeModelContent(() => {
      if (this.changeTimeoutId !== null) {
        clearTimeout(this.changeTimeoutId);
      }
      this.changeTimeoutId = window.setTimeout(() => {
        if (this.onChangeCallback && this.editor) {
          this.onChangeCallback(this.editor.getValue());
        }
      }, 100);
    });
  }

  getValue(): string {
    return this.editor?.getValue() || "";
  }

  setValue(value: string): void {
    this.editor?.setValue(value);
  }

  onChange(callback: (query: string) => void): void {
    this.onChangeCallback = callback;
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
