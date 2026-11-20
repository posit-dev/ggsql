import * as assert from 'assert';
import * as vscode from 'vscode';
import type * as positron from '@posit-dev/positron';
import { parseResultLink, resultKey, withResultLinks } from '../resultLinks';
import { ResultExplorer } from '../dataExplorer/handler';

suite('parseResultLink', () => {
	test('reads the session id and result key of a link', () => {
		assert.deepStrictEqual(
			parseResultLink('x-ggsql-result:ggsql-1a2b/0f9e8d'),
			{ sessionId: 'ggsql-1a2b', key: '0f9e8d' }
		);
	});

	test('ignores other schemes, the old one included', () => {
		assert.strictEqual(parseResultLink('x-r-help:topic'), undefined);
		assert.strictEqual(parseResultLink('https://ggsql.org'), undefined);
		assert.strictEqual(parseResultLink('x-ggsql-view:s/abc'), undefined);
	});

	test('rejects a key the kernel could not have produced', () => {
		assert.strictEqual(parseResultLink('x-ggsql-result:s/abc%0ASELECT%201'), undefined);
		assert.strictEqual(parseResultLink('x-ggsql-result:s/ABC'), undefined);
		assert.strictEqual(parseResultLink('x-ggsql-result:s'), undefined);
		assert.strictEqual(parseResultLink('x-ggsql-result:s/abc/def'), undefined);
	});
});

type Message = positron.LanguageRuntimeMessage & Record<string, unknown>;

function message(id: string, type: string, extra: Record<string, unknown> = {}): Message {
	return {
		id,
		parent_id: 'exec-1',
		when: '2026-01-01T00:00:00Z',
		type: type as positron.LanguageRuntimeMessageType,
		...extra,
	};
}

function result(id: string, key?: unknown): Message {
	return message(id, 'result', {
		data: { 'text/plain': 'table' },
		execution_count: 1,
		outputMetadata: key === undefined ? {} : { ggsql: { result: key } },
	});
}

/** A stand-in supervisor session, with the received messages under `fire`. */
function fakeSession(sessionMode: string) {
	const messages = new vscode.EventEmitter<positron.LanguageRuntimeMessage>();
	const ended = new vscode.EventEmitter<positron.LanguageRuntimeExit>();
	const state = new vscode.EventEmitter<positron.RuntimeState>();
	const calls: string[] = [];
	let disposed = false;
	const session = {
		metadata: { sessionId: 'ggsql-abc', sessionMode },
		onDidReceiveRuntimeMessage: messages.event,
		onDidEndSession: ended.event,
		onDidChangeRuntimeState: state.event,
		callMethod: async (method: string) => {
			calls.push(method);
			return method === 'result_open'
				? { view: 'v1', key: '0f9e', title: 'Query [1]', query: '', columns: [], num_rows: 1 }
				: {};
		},
		dispose() {
			disposed = true;
		},
		whoAmI() {
			return this;
		},
	};
	return {
		session: session as unknown as positron.LanguageRuntimeSession,
		raw: session,
		fire: (m: positron.LanguageRuntimeMessage) => messages.fire(m),
		end: () => ended.fire({ reason: 'restart' } as positron.LanguageRuntimeExit),
		setState: (s: string) => state.fire(s as positron.RuntimeState),
		calls,
		disposed: () => disposed,
	};
}

const tick = () => new Promise(resolve => setTimeout(resolve, 0));

function received(sessionMode: string, input: Message[]): Message[] {
	const fake = fakeSession(sessionMode);
	const wrapped = withResultLinks(fake.session, new ResultExplorer(() => { }));
	const out: Message[] = [];
	wrapped.onDidReceiveRuntimeMessage(m => out.push(m as Message));
	input.forEach(fake.fire);
	wrapped.dispose();
	return out;
}

suite('withResultLinks', () => {
	test('follows a marked result with a link to it, in a console', () => {
		const r = result('r1', '0f9e');
		const out = received('console', [message('i1', 'input'), r, message('s1', 'state')]);
		assert.deepStrictEqual(out.map(m => m.id), ['i1', 'r1', 'r1-ggsql-result-link', 's1']);
		assert.strictEqual(out[1], r);
		const link = out[2];
		assert.strictEqual(link.type, 'stream');
		assert.strictEqual(link.name, 'stdout');
		assert.strictEqual(link.parent_id, 'exec-1');
		assert.strictEqual(link.when, r.when);
		assert.strictEqual(
			link.text,
			'\x1b]8;;x-ggsql-result:ggsql-abc/0f9e\x1b\\Open in Data Explorer\x1b]8;;\x1b\\\n'
		);
	});

	test('adds nothing outside a console', () => {
		for (const mode of ['notebook', 'background']) {
			const out = received(mode, [result('r1', '0f9e')]);
			assert.deepStrictEqual(out.map(m => m.id), ['r1'], mode);
		}
	});

	test('adds nothing for a result without a valid key', () => {
		const out = received('console', [
			result('r1'),
			result('r2', 'NOT HEX'),
			result('r3', 42),
			message('o1', 'output', { outputMetadata: { ggsql: { result: '0f9e' } } }),
		]);
		assert.deepStrictEqual(out.map(m => m.id), ['r1', 'r2', 'r3', 'o1']);
	});

	test('passes everything else through, bound to the session', async () => {
		const fake = fakeSession('console');
		const wrapped = withResultLinks(fake.session, new ResultExplorer(() => { }));
		assert.strictEqual(wrapped.metadata.sessionId, 'ggsql-abc');
		assert.strictEqual((wrapped as unknown as typeof fake.raw).whoAmI(), fake.raw);
		assert.strictEqual(await wrapped.openResource!('https://ggsql.org'), false);
		wrapped.dispose();
		assert.ok(fake.disposed());
	});

	test('clicking a link opens a view and shows it under the result\'s title', async () => {
		const fake = fakeSession('console');
		const shown: object[] = [];
		const wrapped = withResultLinks(fake.session, new ResultExplorer(() => { }), async options => {
			shown.push(options);
		});
		assert.strictEqual(await wrapped.openResource!('x-ggsql-result:ggsql-abc/0f9e'), true);
		await tick();
		assert.deepStrictEqual(fake.calls, ['result_open']);
		assert.deepStrictEqual(shown, [{ providerId: 'ggsql', datasetId: 'ggsql:ggsql-abc/0f9e', displayName: 'Query [1]' }]);
		wrapped.dispose();
	});

	test('a view whose Data Explorer fails to open is closed again', async () => {
		const fake = fakeSession('console');
		const wrapped = withResultLinks(fake.session, new ResultExplorer(() => { }), async () => {
			throw new Error('no editor');
		});
		await wrapped.openResource!('x-ggsql-result:ggsql-abc/0f9e');
		await tick();
		await tick();
		assert.deepStrictEqual(fake.calls, ['result_open', 'result_close']);
		wrapped.dispose();
	});

	test('a failed show leaves an already open Data Explorer\'s view alone', async () => {
		const fake = fakeSession('console');
		let fail = false;
		const wrapped = withResultLinks(fake.session, new ResultExplorer(() => { }), async () => {
			if (fail) {
				throw new Error('already open');
			}
		});
		await wrapped.openResource!('x-ggsql-result:ggsql-abc/0f9e');
		await tick();
		fail = true;
		await wrapped.openResource!('x-ggsql-result:ggsql-abc/0f9e');
		await tick();
		await tick();
		assert.deepStrictEqual(fake.calls, ['result_open']);
		wrapped.dispose();
	});

	test('a link reopens a result whose Data Explorer was closed', async () => {
		const fake = fakeSession('console');
		const explorer = new ResultExplorer(() => { });
		const wrapped = withResultLinks(fake.session, explorer, async () => { });
		await wrapped.openResource!('x-ggsql-result:ggsql-abc/0f9e');
		await tick();
		explorer.closeDataset('ggsql:ggsql-abc/0f9e');
		await tick();
		await wrapped.openResource!('x-ggsql-result:ggsql-abc/0f9e');
		await tick();
		assert.deepStrictEqual(fake.calls, ['result_open', 'result_close', 'result_open']);
		wrapped.dispose();
	});

	test('a session that ended answers at once, and again once it is ready', async () => {
		const fake = fakeSession('console');
		const events: string[] = [];
		const explorer = new ResultExplorer(e => events.push(e.method));
		const wrapped = withResultLinks(fake.session, explorer, async () => { });
		const getState = () => explorer.handleRpc({ method: 'get_state', uri: 'ggsql:ggsql-abc/0f9e', params: {} });
		await getState();

		fake.end();
		assert.deepStrictEqual(events, ['schema_update']);
		const down = (await getState()).result as Record<string, unknown>;
		assert.strictEqual(down.connected, false);
		assert.match(String(down.error_message), /not running/);
		assert.deepStrictEqual(fake.calls, ['result_open']);

		fake.setState('ready');
		assert.deepStrictEqual(events, ['schema_update', 'schema_update']);
		const up = (await getState()).result as Record<string, unknown>;
		assert.notStrictEqual(up.connected, false);
		assert.deepStrictEqual(fake.calls, ['result_open', 'result_open']);
		wrapped.dispose();
	});

	test('resultKey reads only result messages', () => {
		assert.strictEqual(resultKey(result('r', 'ab12')), 'ab12');
		assert.strictEqual(resultKey(message('o', 'output', { outputMetadata: { ggsql: { result: 'ab' } } })), undefined);
	});
});
