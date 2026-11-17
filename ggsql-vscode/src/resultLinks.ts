/*
 * "Open in Data Explorer" links under console query results.
 *
 * In a Positron console the kernel marks each tabular `execute_result` with
 * `{"ggsql": {"result": "<key>"}}` in its metadata, which the supervisor
 * passes on as the Result message's `outputMetadata`. Each ggsql session is
 * wrapped so that, after such a message, its console also receives a line of
 * stdout holding an OSC 8 hyperlink to `x-ggsql-result:<session id>/<key>`.
 *
 * Positron offers a clicked hyperlink to the last active console session of
 * each language via `openResource`, which the supervisor's session does not
 * implement, so the wrapper adds it. Handling the link opens a kernel view of
 * the result through the Data Explorer backend (`dataExplorer/handler.ts`)
 * and shows it with `positron.dataExplorer.open`.
 */

import * as vscode from 'vscode';
import type * as positron from '@posit-dev/positron';
import { getPositronApi } from './positronApi';
import { log } from './extension';
import { PROVIDER_ID, ResultExplorer, datasetId, parseResultRef, resultRef } from './dataExplorer/handler';
import { errorMessage } from './dataExplorer/format';

export const RESULT_SCHEME = 'x-ggsql-result';

/** A link's target, or undefined when the resource is not one of ours. */
export function parseResultLink(resource: vscode.Uri | string): { sessionId: string; key: string } | undefined {
    const uri = typeof resource === 'string' ? vscode.Uri.parse(resource) : resource;
    if (uri.scheme !== RESULT_SCHEME) {
        return undefined;
    }
    return parseResultRef(uri.path);
}

/** The result key a message carries, when it is a Result marked by the kernel. */
export function resultKey(message: positron.LanguageRuntimeMessage): string | undefined {
    if (message.type !== ('result' as positron.LanguageRuntimeMessageType)) {
        return undefined;
    }
    const metadata = (message as positron.LanguageRuntimeResult).outputMetadata;
    const ggsql = metadata?.ggsql as { result?: unknown } | undefined;
    const key = ggsql?.result;
    return typeof key === 'string' && /^[0-9a-f]+$/.test(key) ? key : undefined;
}

/** The stdout line, under a result, that links to it. */
export function linkMessage(
    result: positron.LanguageRuntimeMessage,
    sessionId: string,
    key: string
): positron.LanguageRuntimeStream {
    const target = `${RESULT_SCHEME}:${resultRef(sessionId, key)}`;
    return {
        id: `${result.id}-ggsql-result-link`,
        parent_id: result.parent_id,
        when: result.when,
        type: 'stream' as positron.LanguageRuntimeMessageType,
        name: 'stdout' as positron.LanguageRuntimeStreamName,
        text: `\x1b]8;;${target}\x1b\\Open in Data Explorer\x1b]8;;\x1b\\\n`,
    };
}

/** Open a result in the Data Explorer. */
async function openResult(explorer: ResultExplorer, sessionId: string, key: string): Promise<void> {
    const api = getPositronApi();
    if (!api) {
        return;
    }
    const id = datasetId(sessionId, key);
    try {
        // Opened here rather than on the Data Explorer's first request, for
        // the title, and so an expired result is reported before a tab opens.
        const dataset = await explorer.open(id);
        await api.dataExplorer.open({
            providerId: PROVIDER_ID,
            datasetId: id,
            displayName: dataset.result.title,
        });
    } catch (err) {
        const message = errorMessage(err);
        log(`Opening result ${key} in session ${sessionId} failed: ${message}`);
        vscode.window.showWarningMessage(
            message || 'This query result could not be opened in the Data Explorer.'
        );
    }
}

/**
 * Wrap a supervisor session to add result links to its console and handle
 * `x-ggsql-result:` links.
 *
 * Every runtime message is re-fired in order, with the link line following
 * each marked result in a console session. Everything else passes through to
 * the session itself, with methods bound to it so its own `this` is never the
 * proxy.
 */
export function withResultLinks(
    session: positron.LanguageRuntimeSession,
    explorer: ResultExplorer
): positron.LanguageRuntimeSession {
    const sessionId = session.metadata.sessionId;
    const isConsole = session.metadata.sessionMode === ('console' as positron.LanguageRuntimeSessionMode);

    const messages = new vscode.EventEmitter<positron.LanguageRuntimeMessage>();
    const disposables: { dispose(): unknown }[] = [
        messages,
        explorer.addSession(sessionId, session),
        session.onDidReceiveRuntimeMessage(message => {
            messages.fire(message);
            const key = isConsole ? resultKey(message) : undefined;
            if (key) {
                messages.fire(linkMessage(message, sessionId, key));
            }
        }),
        // The kernel's views end with it.
        session.onDidEndSession(() => explorer.forgetViews(sessionId)),
    ];

    const openResource = async (resource: vscode.Uri | string): Promise<boolean> => {
        const link = parseResultLink(resource);
        if (!link) {
            return false;
        }
        // Positron offers the link only to the most recently active ggsql
        // console, which need not be the one that printed it; the session id
        // routes it back to the right one.
        void openResult(explorer, link.sessionId, link.key);
        return true;
    };

    const dispose = () => {
        disposables.forEach(d => d.dispose());
        return session.dispose();
    };

    return new Proxy(session, {
        get(target, prop) {
            switch (prop) {
                case 'openResource':
                    return openResource;
                case 'onDidReceiveRuntimeMessage':
                    return messages.event;
                case 'dispose':
                    return dispose;
            }
            const value = Reflect.get(target, prop, target);
            return typeof value === 'function' ? value.bind(target) : value;
        },
    });
}
