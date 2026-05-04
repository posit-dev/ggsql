import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import {
    buildPreviewHtml,
    buildRenderArgs,
    getGgsqlCliPath,
    getReaderUri,
} from '../src/preview';

test('buildRenderArgs runs ggsql with file input and Vega-Lite output', () => {
    assert.deepEqual(
        buildRenderArgs('/tmp/query.ggsql', '/tmp/plot.vl.json', 'duckdb://memory'),
        [
            'run',
            '/tmp/query.ggsql',
            '--reader',
            'duckdb://memory',
            '--writer',
            'vegalite',
            '--output',
            '/tmp/plot.vl.json',
        ],
    );
});

test('preview settings fall back to ggsql binary and in-memory DuckDB', () => {
    const emptyConfig = {
        get<T>(_name: string, fallback: T): T {
            return fallback;
        },
    };

    assert.equal(getGgsqlCliPath(emptyConfig), 'ggsql');
    assert.equal(getReaderUri(emptyConfig), 'duckdb://memory');
});

test('buildPreviewHtml embeds a Vega-Lite spec without raw script-breaking text', () => {
    const spec = JSON.stringify({
        title: '</script><img src=x onerror=alert(1)>',
        data: { values: [{ x: 1, y: 2 }] },
        mark: 'point',
        encoding: {
            x: { field: 'x', type: 'quantitative' },
            y: { field: 'y', type: 'quantitative' },
        },
    });

    const html = buildPreviewHtml(spec, {
        cspSource: 'vscode-resource:',
        rendererScriptUri: 'vscode-resource:/out/previewWebview.js',
        nonce: 'test-nonce',
        title: 'ggsql test plot',
    });

    assert.match(html, /script-src 'nonce-test-nonce' vscode-resource:/);
    assert.match(html, /ggsql test plot/);
    assert(!html.includes('</script><img'));
    assert(html.includes('\\u003c/script\\u003e'));
    assert.match(html, /id="ggsql-spec"/);
    assert.match(html, /src="vscode-resource:\/out\/previewWebview\.js"/);
    assert(!html.includes('https://cdn.jsdelivr.net'));
});

test('buildPreviewHtml renders chart-only preview like Markdown preview', () => {
    const html = buildPreviewHtml('{"mark":"point","data":{"values":[]}}', {
        cspSource: 'vscode-resource:',
        rendererScriptUri: 'vscode-resource:/out/previewWebview.js',
        nonce: 'test-nonce',
        title: 'ggsql test plot',
    });

    assert.match(html, /id="chart-view"/);
    assert(!html.includes('data-mode="code"'));
    assert(!html.includes('data-mode="both"'));
    assert(!html.includes('Generated Vega-Lite JSON'));
});

test('extension uses real SVG icon files for top-level render action', () => {
    const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
    const renderCommand = packageJson.contributes.commands.find(
        (command: { command: string }) => command.command === 'ggsql.renderPlot'
    );
    const renderMenu = packageJson.contributes.menus['editor/title'].find(
        (item: { command: string }) => item.command === 'ggsql.renderPlot'
    );

    assert.deepEqual(renderCommand.icon, {
        light: './resources/render-plot-light.svg',
        dark: './resources/render-plot-dark.svg',
    });
    assert.equal(fs.existsSync('resources/render-plot-light.svg'), true);
    assert.equal(fs.existsSync('resources/render-plot-dark.svg'), true);
    assert.equal(renderMenu.group, 'navigation@-1');
});

test('webview renderer uses Vega AST interpreter for CSP-safe rendering', () => {
    const source = fs.readFileSync('src/previewWebview.ts', 'utf8');

    assert.match(source, /ast:\s*true/);
    assert(!source.includes('unsafe-eval'));
});

test('webview renderer fits chart to available preview area', () => {
    const source = fs.readFileSync('src/previewWebview.ts', 'utf8');

    assert.match(source, /getBoundingClientRect/);
    assert.match(source, /autosize:\s*spec\.autosize\s*\?\?/);
    assert.match(source, /type:\s*'fit'/);
    assert.match(source, /resize:\s*true/);
    assert.match(source, /const resolvedWidth = spec\.width \?\? width/);
    assert.match(source, /const resolvedHeight = spec\.height \?\? height/);
    assert.match(source, /width:\s*resolveDimension\(resolvedWidth, width\)/);
    assert.match(source, /height:\s*resolveDimension\(resolvedHeight, height\)/);
});

test('webview renderer resolves Vega-Lite container dimensions to preview pixels', () => {
    const source = fs.readFileSync('src/previewWebview.ts', 'utf8');

    assert.match(source, /resolveDimension/);
    assert.match(source, /dimension === 'container'/);
});

test('render command writes detailed diagnostics to the ggsql output channel', () => {
    const source = fs.readFileSync('src/extension.ts', 'utf8');

    assert.match(source, /outputChannel\.show\(true\)/);
    assert.match(source, /Render Plot clicked/);
    assert.match(source, /Working directory:/);
    assert.match(source, /Query temp file:/);
    assert.match(source, /Vega-Lite output file:/);
    assert.match(source, /ggsql CLI exited/);
    assert.match(source, /Preview panel/);
});
