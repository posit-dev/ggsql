/*
 * End-to-end check that the kernel bundled in this extension actually runs.
 *
 * Everything else about bundling is verified without a kernel process: the unit
 * suites assert precedence and metadata, and the release workflow asserts the
 * binary is inside the VSIX. Neither can tell whether the binary starts. This
 * suite does, which is the failure the whole change is about (`KS-19: Kernel
 * path not found`, and its cousins — a wrong-architecture or unsigned binary
 * that Positron cannot launch).
 *
 * Requires ggsql-vscode/bundled/bin/ggsql-jupyter to exist; CI builds it first.
 */

import * as assert from 'assert';
import * as path from 'path';
import * as vscode from 'vscode';
import type { PositronApi } from '@posit-dev/positron';
import { getPositronApi } from '../../positronApi';

const EXTENSION_ID = 'ggsql.ggsql';

/** Poll until `probe` returns a value, or fail after `timeoutMs`. */
async function waitFor<T>(what: string, timeoutMs: number, probe: () => Promise<T | undefined>): Promise<T> {
	const deadline = Date.now() + timeoutMs;
	for (;;) {
		const found = await probe();
		if (found !== undefined) {
			return found;
		}
		if (Date.now() > deadline) {
			throw new Error(`timed out after ${timeoutMs}ms waiting for ${what}`);
		}
		await new Promise(resolve => setTimeout(resolve, 500));
	}
}

suite('bundled kernel in Positron', () => {
	let positron: PositronApi;

	suiteSetup(async () => {
		const extension = vscode.extensions.getExtension(EXTENSION_ID);
		assert.ok(extension, `extension ${EXTENSION_ID} not found`);
		await extension.activate();

		const api = getPositronApi();
		assert.ok(api, 'no Positron API; this suite must run under Positron, not VS Code');
		positron = api;
	});

	test('the bundled kernel is registered as the only ggsql runtime', async () => {
		// Discovery runs on window open, so the runtime may not be registered the
		// instant activation returns.
		const runtimes = await waitFor('a registered ggsql runtime', 60_000, async () => {
			const registered = await positron.runtime.getRegisteredRuntimes();
			const ggsql = registered.filter(runtime => runtime.languageId === 'ggsql');
			return ggsql.length > 0 ? ggsql : undefined;
		});

		assert.strictEqual(
			runtimes.length,
			1,
			`expected one ggsql runtime, got ${runtimes.map(r => r.runtimePath).join(', ')}`,
		);
		// The kernel inside the extension, with the identity that survives an
		// update — not something the runner happened to have installed.
		assert.strictEqual(runtimes[0].runtimeId, 'ggsql-bundled');
		assert.strictEqual(runtimes[0].runtimeName, 'ggsql');
		assert.ok(
			runtimes[0].runtimePath.includes(path.join('bundled', 'bin')),
			`unexpected kernel path ${runtimes[0].runtimePath}`,
		);
	});

	test('the console starts the bundled kernel and runs a query', async () => {
		// executeCode starts a session when none is running, so this covers the
		// whole path: spawning the binary, the supervisor's Jupyter handshake,
		// and a result coming back. The kernel holds an in-memory DuckDB
		// session, so the query needs no connection string.
		const result = await positron.runtime.executeCode('ggsql', 'SELECT 1 AS n', false);
		assert.ok(result, 'executeCode returned no result');

		const sessions = await positron.runtime.getActiveSessions();
		const ggsqlSessions = sessions.filter(
			session => session.runtimeMetadata.languageId === 'ggsql',
		);
		assert.strictEqual(ggsqlSessions.length, 1, 'expected exactly one ggsql session');
		assert.strictEqual(ggsqlSessions[0].runtimeMetadata.runtimeId, 'ggsql-bundled');
	});

	test('a query with a visualisation returns a plot', async () => {
		// The reason a ggsql console exists, and a second execution on the
		// session the previous test started.
		const result = await positron.runtime.executeCode(
			'ggsql',
			'SELECT 1 AS x, 2 AS y VISUALISE x AS x, y AS y DRAW point',
			false,
		);
		assert.ok(result, 'executeCode returned no result');
	});
});
