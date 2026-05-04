export const DEFAULT_READER_URI = 'duckdb://memory';

interface ConfigReader {
    get<T>(name: string, defaultValue: T): T;
}

interface PreviewHtmlOptions {
    cspSource: string;
    rendererScriptUri: string;
    nonce?: string;
    title?: string;
}

export function getGgsqlCliPath(config: ConfigReader): string {
    const configuredPath = config.get<string>('cliPath', '').trim();
    return configuredPath || 'ggsql';
}

export function getReaderUri(config: ConfigReader): string {
    const readerUri = config.get<string>('readerUri', DEFAULT_READER_URI).trim();
    return readerUri || DEFAULT_READER_URI;
}

export function buildRenderArgs(queryFile: string, outputFile: string, readerUri: string): string[] {
    return [
        'run',
        queryFile,
        '--reader',
        readerUri,
        '--writer',
        'vegalite',
        '--output',
        outputFile,
    ];
}

export function buildPreviewHtml(specJson: string, options: PreviewHtmlOptions): string {
    const parsedSpec = JSON.parse(specJson) as unknown;
    const title = options.title ?? 'ggsql Plot';
    const nonce = options.nonce ?? createNonce();
    const encodedSpec = escapeForScript(JSON.stringify(parsedSpec));
    const safeTitle = escapeHtml(title);

    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${options.cspSource} data:; style-src ${options.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}' ${options.cspSource};">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${safeTitle}</title>
    <style>
        :root {
            color-scheme: light dark;
        }

        body {
            margin: 0;
            background: var(--vscode-editor-background);
            color: var(--vscode-editor-foreground);
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
        }

        .preview-shell {
            display: flex;
            flex-direction: column;
            height: 100vh;
            min-height: 0;
            overflow: hidden;
        }

        .preview-header {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 12px;
            border-bottom: 1px solid var(--vscode-panel-border);
            background: var(--vscode-sideBar-background);
        }

        h1 {
            flex: 1 1 auto;
            margin: 0;
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-size: 13px;
            font-weight: 600;
        }

        #chart-view {
            flex: 1 1 auto;
            min-height: 0;
            min-width: 0;
            box-sizing: border-box;
            overflow: auto;
            padding: 16px;
        }

        #vis {
            width: 100%;
            height: 100%;
            min-height: 320px;
        }

        .vega-embed,
        .vega-embed > svg,
        .vega-embed > canvas {
            max-width: 100%;
        }

        .error {
            display: none;
            margin: 12px 0 0;
            color: var(--vscode-errorForeground);
            white-space: pre-wrap;
            font-family: var(--vscode-editor-font-family);
        }

        @media (max-width: 720px) {
            #chart-view {
                padding: 10px;
            }
        }
    </style>
</head>
<body>
    <div class="preview-shell">
        <header class="preview-header">
            <h1 title="${safeTitle}">${safeTitle}</h1>
        </header>
        <section id="chart-view" aria-label="Rendered chart">
            <div id="vis"></div>
            <pre id="render-error" class="error"></pre>
        </section>
    </div>
    <script type="application/json" id="ggsql-spec">${encodedSpec}</script>
    <script nonce="${nonce}" src="${options.rendererScriptUri}"></script>
</body>
</html>`;
}

function createNonce(): string {
    const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let nonce = '';
    for (let i = 0; i < 32; i++) {
        nonce += alphabet.charAt(Math.floor(Math.random() * alphabet.length));
    }
    return nonce;
}

function escapeForScript(value: string): string {
    return value
        .replace(/</g, '\\u003c')
        .replace(/>/g, '\\u003e')
        .replace(/&/g, '\\u0026')
        .replace(/\u2028/g, '\\u2028')
        .replace(/\u2029/g, '\\u2029');
}

function escapeHtml(value: string): string {
    return value
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}
