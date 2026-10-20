/*
 * "Open in Data Explorer" links under console query results.
 *
 * In a Positron console the kernel follows each tabular result with an OSC 8
 * hyperlink to `x-ggsql-view:<session-id>/<result-key>`. Positron offers a
 * clicked hyperlink to the last active console session of each language via
 * `openResource`, which the supervisor's session does not implement — so each
 * ggsql session is wrapped to add it. Handling the link runs `-- @view-result <key>`
 * silently in the session that printed it, and the kernel opens the result in
 * the data explorer.
 */

import * as vscode from 'vscode';
import type * as positron from '@posit-dev/positron';
import { getPositronApi } from './positronApi';
import { log } from './extension';

export const VIEW_SCHEME = 'x-ggsql-view';

/** A link's target, or undefined when the resource is not one of ours. */
export function parseViewLink(resource: vscode.Uri | string): { sessionId: string; key: string } | undefined {
    const uri = typeof resource === 'string' ? vscode.Uri.parse(resource) : resource;
    if (uri.scheme !== VIEW_SCHEME) {
        return undefined;
    }
    const [sessionId, key, ...rest] = uri.path.split('/');
    // The key goes into code the kernel executes, so only accept the shape the
    // kernel produces.
    if (!sessionId || !key || rest.length > 0 || !/^[0-9a-f]+$/.test(key)) {
        return undefined;
    }
    return { sessionId, key };
}

/** Open a result in the data explorer by running `-- @view-result` in its session. */
async function openResult(sessionId: string, key: string): Promise<void> {
    const api = getPositronApi();
    if (!api) {
        return;
    }
    try {
        await api.runtime.executeCode(
            'ggsql',
            `-- @view-result ${key}`,
            false,
            false,
            api.RuntimeCodeExecutionMode.Silent,
            api.RuntimeErrorBehavior.Continue,
            undefined,
            sessionId
        );
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        log(`Opening result ${key} in session ${sessionId} failed: ${message}`);
        vscode.window.showWarningMessage(
            message || 'This query result could not be opened in the Data Explorer.'
        );
    }
}

/**
 * Wrap a supervisor session so it handles `x-ggsql-view:` links.
 *
 * Everything else passes through to the session itself, with methods bound to
 * it so its own `this` is never the proxy.
 */
export function withResultLinks(
    session: positron.LanguageRuntimeSession
): positron.LanguageRuntimeSession {
    const openResource = async (resource: vscode.Uri | string): Promise<boolean> => {
        const link = parseViewLink(resource);
        if (!link) {
            return false;
        }
        // Positron offers the link only to the most recently active ggsql
        // console, which need not be the one that printed it; the session id
        // routes it back to the right one.
        void openResult(link.sessionId, link.key);
        return true;
    };

    return new Proxy(session, {
        get(target, prop) {
            if (prop === 'openResource') {
                return openResource;
            }
            const value = Reflect.get(target, prop, target);
            return typeof value === 'function' ? value.bind(target) : value;
        },
    });
}
