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

	test('the bundled kernel is registered, named for the version it reports', async () => {
		// Discovery runs on window open, so the runtime may not be registered the
		// instant activation returns.
		const runtimes = await waitFor('a registered ggsql runtime', 60_000, async () => {
			const registered = await positron.runtime.getRegisteredRuntimes();
			const ggsql = registered.filter(runtime => runtime.languageId === 'ggsql');
			return ggsql.length > 0 ? ggsql : undefined;
		});

		// Every ggsql install on the machine is offered, so this asserts about
		// the bundled one rather than the size of the list — a developer running
		// the suite locally has ggsql installed as well.
		const bundled = runtimes.filter(runtime => runtime.runtimeId === 'ggsql-bundled');
		assert.strictEqual(
			bundled.length,
			1,
			`expected one bundled ggsql runtime, got ${runtimes.map(r => r.runtimePath).join(', ')}`,
		);
		assert.ok(
			bundled[0].runtimePath.includes(path.join('bundled', 'bin')),
			`unexpected kernel path ${bundled[0].runtimePath}`,
		);
		// The version comes from running the binary, so this is the one place
		// the interpolation is checked against a real kernel.
		assert.match(bundled[0].runtimeName, /^ggsql \d+\.\d+\.\d+/);
	});

	test('the console starts the bundled kernel and runs a query', async () => {
		// Starting by runtime id, rather than letting Positron choose one for the
		// language, is what makes this a test of the kernel inside the VSIX:
		// every ggsql install is offered now, so the default is not necessarily
		// the bundled one.
		const session = await positron.runtime.startLanguageRuntime('ggsql-bundled', 'ggsql');
		assert.strictEqual(session.runtimeMetadata.runtimeId, 'ggsql-bundled');

		// executeCode goes to that session, so this covers the whole path:
		// spawning the binary, the supervisor's Jupyter handshake, and a result
		// coming back. The kernel holds an in-memory DuckDB session, so the
		// query needs no connection string.
		const result = await positron.runtime.executeCode('ggsql', 'SELECT 1 AS n', false);
		assert.ok(result, 'executeCode returned no result');
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
