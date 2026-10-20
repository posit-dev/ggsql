import * as assert from 'assert';
import { parseViewLink } from '../resultLinks';

suite('parseViewLink', () => {
	test('reads the session id and result key the kernel writes', () => {
		assert.deepStrictEqual(
			parseViewLink('x-ggsql-view:ggsql-1a2b/0f9e8d'),
			{ sessionId: 'ggsql-1a2b', key: '0f9e8d' }
		);
	});

	test('ignores other schemes', () => {
		assert.strictEqual(parseViewLink('x-r-help:topic'), undefined);
		assert.strictEqual(parseViewLink('https://ggsql.org'), undefined);
	});

	test('rejects a key the kernel could not have produced', () => {
		// The key is spliced into code the kernel runs.
		assert.strictEqual(parseViewLink('x-ggsql-view:s/abc%0ASELECT%201'), undefined);
		assert.strictEqual(parseViewLink('x-ggsql-view:s/ABC'), undefined);
		assert.strictEqual(parseViewLink('x-ggsql-view:s'), undefined);
		assert.strictEqual(parseViewLink('x-ggsql-view:s/abc/def'), undefined);
	});
});
