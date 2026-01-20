/*
 * ggsql VS Code Extension
 *
 * Provides syntax highlighting for ggsql and, when running in Positron,
 * a language runtime that wraps the ggsql-jupyter kernel.
 */

import * as vscode from 'vscode';
import { tryAcquirePositronApi } from '@posit-dev/positron';
import { GgsqlRuntimeManager } from './manager';

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
}

/**
 * Deactivates the extension.
 */
export function deactivate(): void {
    // Nothing to clean up
}
