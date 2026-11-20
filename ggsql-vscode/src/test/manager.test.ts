import * as assert from 'assert';
import * as vscode from 'vscode';
import { GgsqlRuntimeManager, MAX_CONSOLE_ROWS, consoleRowsEnv, createDynState, getSupervisorApi, keepResultsEnv, resultsMemoryEnv } from '../manager';

suite('manager', () => {
	// The suites run in stock VS Code, where positron.positron-supervisor is
	// not installed. All three call sites depend on this rejecting rather than
	// resolving with a partially usable object.
	test('getSupervisorApi rejects when the supervisor is absent', async () => {
		await assert.rejects(
			() => getSupervisorApi(),
			/Positron Supervisor extension not found/,
		);
	});

	test('resultsMemoryEnv passes a byte count the kernel accepts', () => {
		assert.deepStrictEqual(resultsMemoryEnv(' 512MB '), { GGSQL_RESULTS_MAX_BYTES: '512MB' });
		assert.deepStrictEqual(resultsMemoryEnv('2gb'), { GGSQL_RESULTS_MAX_BYTES: '2gb' });
		assert.deepStrictEqual(resultsMemoryEnv('1048576'), { GGSQL_RESULTS_MAX_BYTES: '1048576' });
	});

	test('resultsMemoryEnv leaves out a value the kernel would refuse', () => {
		for (const value of ['', '0', '0gb', '1.5gb', '2 terabytes', '-1mb', '1234567890123gb', 512, undefined]) {
			assert.deepStrictEqual(resultsMemoryEnv(value), {}, String(value));
		}
	});

	test('consoleRowsEnv passes a row count the kernel accepts', () => {
		assert.deepStrictEqual(consoleRowsEnv(25), { GGSQL_CONSOLE_ROWS: '25' });
	});

	test('consoleRowsEnv leaves out a value the kernel would refuse', () => {
		// The kernel exits on a bad value, so the session would fail to start.
		for (const rows of [0, -1, 2.5, '10', undefined, null, MAX_CONSOLE_ROWS + 1, 1e21]) {
			assert.deepStrictEqual(consoleRowsEnv(rows), {}, String(rows));
		}
		assert.deepStrictEqual(consoleRowsEnv(MAX_CONSOLE_ROWS), { GGSQL_CONSOLE_ROWS: String(MAX_CONSOLE_ROWS) });
	});

	test('the setting\'s maximum is the kernel\'s', () => {
		const pkg = require('../../package.json');
		assert.strictEqual(pkg.contributes.configuration.properties['ggsql.consoleRows'].maximum, MAX_CONSOLE_ROWS);
	});

	test('keepResultsEnv turns keeping results on only when asked', () => {
		assert.deepStrictEqual(keepResultsEnv(true), { GGSQL_KEEP_RESULTS: '1' });
		assert.deepStrictEqual(keepResultsEnv(false), {});
	});

	test('createDynState falls back to the default session name', () => {
		const state = createDynState();
		assert.strictEqual(state.sessionName, 'ggsql');
		assert.strictEqual(state.inputPrompt, 'ggsql> ');
		assert.strictEqual(state.continuationPrompt, '... ');
	});

	test('createDynState keeps a name Positron supplies', () => {
		// Positron passes the current session name to restoreSession. Dropping
		// it renames the console back to 'ggsql' on every window reload.
		const state = createDynState('Sales analysis');
		assert.strictEqual(state.sessionName, 'Sales analysis');
	});

	test('createDynState falls back when the supplied name is empty', () => {
		// A blank name would otherwise leave the restored console with no title.
		assert.strictEqual(createDynState('').sessionName, 'ggsql');
	});

	test('the manager opts out of the discovery cache fast path', () => {
		// ggsql runtimes are never marked cacheable, because the kernel path
		// can come from a workspace setting or from PATH. Without this flag
		// Positron would be free to skip discovery on a warm start and leave
		// ggsql unregistered.
		const manager = new GgsqlRuntimeManager({} as vscode.ExtensionContext);
		assert.strictEqual(manager.alwaysRediscover, true);
	});

	test('restoreSession propagates a missing supervisor', async () => {
		// getSupervisorApi() is awaited first, before any kernel spec is
		// written, so the rejection arrives with nothing done on disk.
		const manager = new GgsqlRuntimeManager({} as vscode.ExtensionContext);
		await assert.rejects(
			() => manager.restoreSession({} as never, {} as never, 'Sales analysis'),
			/Positron Supervisor extension not found/,
		);
	});
});
