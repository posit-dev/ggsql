/*
 * ggsql VS Code Extension
 *
 * Provides syntax highlighting for ggsql and, when running in Positron,
 * a language runtime that wraps the ggsql-jupyter kernel.
 */

import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import { promisify } from 'util';
import { tryAcquirePositronApi } from '@posit-dev/positron';
import { GgsqlRuntimeManager } from './manager';
import { createConnectionDrivers } from './connections';
import { GgsqlCodeLensProvider, registerCellCommands } from './codelens';
import { activateDecorations } from './decorations';
import { activateContextKeys } from './context';
import { parseCells } from './cellParser';
import {
    buildPreviewHtml,
    buildRenderArgs,
    getGgsqlCliPath,
    getReaderUri,
} from './preview';

const execFile = promisify(cp.execFile);
const INSTALL_URL = 'https://ggsql.org/get_started/installation.html';
const previewPanels = new Map<string, vscode.WebviewPanel>();

// Output channel for logging
const outputChannel = vscode.window.createOutputChannel('ggsql');

export function log(message: string): void {
    outputChannel.appendLine(`[${new Date().toISOString()}] ${message}`);
}

/**
 * Activates the extension.
 *
 * @param context The extension context
 */
export function activate(context: vscode.ExtensionContext): void {
    log('ggsql extension activating...');

    // Try to acquire the Positron API
    const positronApi = tryAcquirePositronApi();

    context.subscriptions.push(
        vscode.commands.registerCommand('ggsql.renderPlot', async () => {
            await renderPlot(context);
        })
    );

    void checkCliAvailability();

    if (!positronApi) {
        // Running in VS Code (not Positron) - syntax highlighting still works
        // but we don't register the language runtime
        log('Positron API not available - running in VS Code mode');
        return;
    }

    log('Positron API acquired - registering runtime manager');

    // Running in Positron - register the ggsql runtime manager
    const manager = new GgsqlRuntimeManager(context);
    const disposable = positronApi.runtime.registerLanguageRuntimeManager('ggsql', manager);
    context.subscriptions.push(disposable);

    log('ggsql runtime manager registered successfully');

    // Register connection drivers for the Connections pane
    const drivers = createConnectionDrivers(positronApi);
    for (const driver of drivers) {
        const driverDisposable = positronApi.connections.registerConnectionDriver(driver);
        context.subscriptions.push(driverDisposable);
    }

    log(`Registered ${drivers.length} connection drivers`);

    // Register "Source Current File" command for the editor run button
    context.subscriptions.push(
        vscode.commands.registerCommand('ggsql.sourceCurrentFile', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.document.languageId !== 'ggsql') {
                return;
            }
            const cells = parseCells(editor.document);
            if (cells.length > 0) {
                for (const cell of cells) {
                    if (cell.text.length > 0) {
                        positronApi.runtime.executeCode('ggsql', cell.text, false, true);
                    }
                }
            } else {
                const code = editor.document.getText();
                if (code.trim().length === 0) {
                    return;
                }
                positronApi.runtime.executeCode('ggsql', code, true);
            }
        })
    );

    // Register code lens provider and cell commands
    context.subscriptions.push(
        vscode.languages.registerCodeLensProvider('ggsql', new GgsqlCodeLensProvider()),
    );
    registerCellCommands(context, (code) => {
        positronApi.runtime.executeCode('ggsql', code, false, true);
    });

    activateDecorations(context.subscriptions);
    activateContextKeys(context.subscriptions);
}

/**
 * Deactivates the extension.
 */
export function deactivate(): void {
    // Nothing to clean up
}

async function renderPlot(context: vscode.ExtensionContext): Promise<void> {
    outputChannel.show(true);
    log('Render Plot clicked');

    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'ggsql') {
        log('Render Plot skipped: no active ggsql editor');
        return;
    }

    const query = getPreviewQuery(editor);
    if (query.trim().length === 0) {
        log('Render Plot skipped: query is empty');
        return;
    }

    log(`Document: ${editor.document.uri.toString()}`);
    log(`Query source: ${editor.selection.isEmpty ? 'current file' : 'selection'}`);
    log(`Query size: ${query.length} chars`);

    await vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            title: 'Rendering ggsql plot',
            cancellable: false,
        },
        async () => {
            const config = vscode.workspace.getConfiguration('ggsql');
            const cliPath = getGgsqlCliPath(config);
            const readerUri = getReaderUri(config);
            const previewDir = path.join(context.globalStorageUri.fsPath, 'preview');
            const runId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
            const queryPath = path.join(previewDir, `${runId}.ggsql`);
            const outputPath = path.join(previewDir, `${runId}.vl.json`);
            const cwd = getDocumentWorkingDirectory(editor.document);

            log(`CLI path: ${cliPath}`);
            log(`Reader URI: ${readerUri}`);
            log(`Working directory: ${cwd ?? '(none)'}`);
            log(`Preview directory: ${previewDir}`);
            log(`Query temp file: ${queryPath}`);
            log(`Vega-Lite output file: ${outputPath}`);

            await fs.promises.mkdir(previewDir, { recursive: true });
            await fs.promises.writeFile(queryPath, query, 'utf8');
            log('Wrote query temp file');

            const args = buildRenderArgs(queryPath, outputPath, readerUri);
            log(`Running command: ${formatCommand(cliPath, args)}`);
            const startedAt = Date.now();

            try {
                const result = await execFile(cliPath, args, {
                    cwd,
                    timeout: 120_000,
                    maxBuffer: 10 * 1024 * 1024,
                });
                log(`ggsql CLI exited in ${Date.now() - startedAt}ms`);
                if (result.stdout.trim()) {
                    log(`stdout:\n${result.stdout.trim()}`);
                }
                if (result.stderr.trim()) {
                    log(`stderr:\n${result.stderr.trim()}`);
                }
            } catch (error) {
                log(`ggsql CLI failed after ${Date.now() - startedAt}ms`);
                logRenderError(error);
                vscode.window.showErrorMessage(renderErrorMessage(error, cliPath));
                return;
            }

            let specJson: string;
            try {
                specJson = await fs.promises.readFile(outputPath, 'utf8');
            } catch (error) {
                logRenderError(error);
                vscode.window.showErrorMessage(`ggsql did not write a Vega-Lite spec: ${errorMessage(error)}`);
                return;
            }

            log(`Read Vega-Lite spec (${Buffer.byteLength(specJson, 'utf8')} bytes)`);
            showPreview(context, editor.document, specJson);
        }
    );
}

function getPreviewQuery(editor: vscode.TextEditor): string {
    if (!editor.selection.isEmpty) {
        return editor.document.getText(editor.selection);
    }
    return editor.document.getText();
}

function getDocumentWorkingDirectory(document: vscode.TextDocument): string | undefined {
    const workspaceFolder = vscode.workspace.getWorkspaceFolder(document.uri);
    if (workspaceFolder) {
        return workspaceFolder.uri.fsPath;
    }

    if (document.uri.scheme === 'file') {
        return path.dirname(document.uri.fsPath);
    }

    return undefined;
}

function showPreview(context: vscode.ExtensionContext, document: vscode.TextDocument, specJson: string): void {
    const key = document.uri.toString();
    const title = `Preview ${path.basename(document.fileName || 'plot')}`;
    const existingPanel = previewPanels.get(key);
    if (existingPanel) {
        log(`Preview panel reused: ${title}`);
        existingPanel.reveal(vscode.ViewColumn.Beside);
        updatePreviewHtml(context, existingPanel, title, specJson);
        return;
    }

    log(`Preview panel created: ${title}`);
    const panel = vscode.window.createWebviewPanel(
        'ggsql.plotPreview',
        title,
        vscode.ViewColumn.Beside,
        {
            enableScripts: true,
            retainContextWhenHidden: true,
        }
    );
    previewPanels.set(key, panel);
    panel.onDidDispose(() => {
        previewPanels.delete(key);
    });
    updatePreviewHtml(context, panel, title, specJson);
}

function updatePreviewHtml(
    context: vscode.ExtensionContext,
    panel: vscode.WebviewPanel,
    title: string,
    specJson: string
): void {
    try {
        const rendererScriptUri = panel.webview.asWebviewUri(
            vscode.Uri.joinPath(context.extensionUri, 'out', 'previewWebview.js')
        );
        panel.webview.html = buildPreviewHtml(specJson, {
            cspSource: panel.webview.cspSource,
            rendererScriptUri: rendererScriptUri.toString(),
            title,
        });
        log(`Preview HTML updated: ${title}`);
    } catch (error) {
        panel.dispose();
        logRenderError(error);
        vscode.window.showErrorMessage(`Failed to open ggsql preview: ${errorMessage(error)}`);
    }
}

function renderErrorMessage(error: unknown, cliPath: string): string {
    const err = error as NodeJS.ErrnoException & { stderr?: string };
    if (err.code === 'ENOENT') {
        void showMissingCliMessage(cliPath);
        return `Could not find ggsql CLI at '${cliPath}'. Install ggsql, set ggsql.cliPath, or add ggsql to PATH.`;
    }

    const stderr = typeof err.stderr === 'string' ? err.stderr.trim() : '';
    return stderr || errorMessage(error);
}

function logRenderError(error: unknown): void {
    log(errorMessage(error));
    const err = error as { stderr?: string; stdout?: string };
    if (typeof err.stdout === 'string' && err.stdout.trim()) {
        log(err.stdout.trim());
    }
    if (typeof err.stderr === 'string' && err.stderr.trim()) {
        log(err.stderr.trim());
    }
}

function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

function formatCommand(command: string, args: string[]): string {
    return [command, ...args].map(formatCommandArg).join(' ');
}

function formatCommandArg(arg: string): string {
    if (/^[A-Za-z0-9_./:=@-]+$/.test(arg)) {
        return arg;
    }

    return JSON.stringify(arg);
}

async function checkCliAvailability(): Promise<void> {
    const config = vscode.workspace.getConfiguration('ggsql');
    const cliPath = getGgsqlCliPath(config);
    try {
        const result = await execFile(cliPath, ['--version'], {
            timeout: 5000,
            maxBuffer: 1024 * 1024,
        });
        const version = result.stdout.trim() || result.stderr.trim();
        if (version) {
            log(`Found ggsql CLI: ${version}`);
        }
    } catch (error) {
        const err = error as NodeJS.ErrnoException;
        if (err.code === 'ENOENT') {
            await showMissingCliMessage(cliPath);
        } else {
            log(`Could not verify ggsql CLI '${cliPath}': ${errorMessage(error)}`);
        }
    }
}

async function showMissingCliMessage(cliPath: string): Promise<void> {
    const selection = await vscode.window.showWarningMessage(
        `ggsql CLI not found at '${cliPath}'. Render Plot needs the ggsql CLI.`,
        'Install ggsql',
        'Set CLI Path'
    );

    if (selection === 'Install ggsql') {
        await vscode.env.openExternal(vscode.Uri.parse(INSTALL_URL));
    } else if (selection === 'Set CLI Path') {
        await vscode.commands.executeCommand('workbench.action.openSettings', 'ggsql.cliPath');
    }
}
