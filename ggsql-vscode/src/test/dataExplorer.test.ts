import * as assert from 'assert';
import type * as positron from '@posit-dev/positron';
import { ResultExplorer, datasetId, parseDatasetId } from '../dataExplorer/handler';
import { searchSchema, toKernelFilter } from '../dataExplorer/filters';
import { toProfileCalls, toSummaryStats } from '../dataExplorer/profiles';
import { cellValue } from '../dataExplorer/format';
import type { KernelColumn, OpenedResult, RowFilter } from '../dataExplorer/protocol';

const COLUMNS: KernelColumn[] = [
	{ name: 'n', type_name: 'BIGINT', kind: 'integer' },
	{ name: 'x', type_name: 'DOUBLE', kind: 'floating' },
	{ name: 'Label', type_name: 'VARCHAR', kind: 'string' },
];

const OPENED: OpenedResult = {
	view: 'v1',
	key: '0f9e',
	title: 'Query [3]',
	query: 'SELECT 1',
	columns: COLUMNS,
	num_rows: 100,
};

const ID = 'ggsql:ggsql-abc/0f9e';

/** A results API stand-in that records each call and answers from `replies`. */
function setup(replies: Record<string, (params: Record<string, unknown>) => unknown> = {}) {
	const calls: Array<[string, Record<string, unknown>]> = [];
	const events: positron.DataExplorerUiEvent[] = [];
	const explorer = new ResultExplorer(e => events.push(e));
	explorer.addSession('ggsql-abc', {
		callMethod: async (method: string, params: Record<string, unknown>) => {
			calls.push([method, params]);
			if (method === 'result_open') {
				return OPENED;
			}
			const reply = replies[method];
			if (!reply) {
				throw { code: -32000, message: `no reply for ${method}` };
			}
			return reply(params);
		},
	});
	const rpc = (method: string, params: object = {}) =>
		explorer.handleRpc({ method, uri: ID, params });
	return { explorer, calls, events, rpc };
}

function filter(filter_type: string, column_index: number, params?: object): RowFilter {
	return {
		filter_id: `f-${filter_type}`,
		filter_type: filter_type as RowFilter['filter_type'],
		column_schema: { column_name: '', column_index, type_name: '', type_display: 'integer' },
		condition: 'and',
		params: params as RowFilter['params'],
	};
}

suite('Data Explorer backend', () => {
	test('dataset ids are a session id and a hex key', () => {
		assert.deepStrictEqual(parseDatasetId('ggsql:s/0a'), { sessionId: 's', key: '0a' });
		assert.strictEqual(parseDatasetId('s/0a'), undefined);
		assert.strictEqual(parseDatasetId('ggsql:s/0A'), undefined);
		assert.strictEqual(parseDatasetId('ggsql:s/0a/1'), undefined);
		assert.strictEqual(datasetId('ggsql-abc', '0f9e'), ID);
		// What Positron's PositronDataExplorerUri.parse accepts besides a UUID.
		assert.match(datasetId('ggsql-abc', '0f9e'), /^[a-z][a-z0-9+.-]*:.+$/);
	});

	test('opens one kernel view per dataset, lazily', async () => {
		const { calls, rpc } = setup();
		const state = await rpc('get_state');
		await rpc('get_schema', { column_indices: [2, 0, 9] });
		assert.deepStrictEqual(calls, [['result_open', { key: '0f9e' }]]);
		const s = state.result as Record<string, any>;
		assert.strictEqual(s.display_name, 'Query [3]');
		assert.deepStrictEqual(s.table_shape, { num_rows: 100, num_columns: 3 });
		assert.strictEqual(s.supported_features.set_row_filters.supports_conditions, 'unsupported');
		assert.strictEqual(s.supported_features.set_column_filters.support_status, 'unsupported');
	});

	test('get_schema describes the asked-for columns', async () => {
		const { rpc } = setup();
		const r = await rpc('get_schema', { column_indices: [2, 9] });
		assert.deepStrictEqual(r.result, {
			columns: [{ column_name: 'Label', column_index: 2, type_name: 'VARCHAR', type_display: 'string' }],
		});
	});

	test('a result that cannot be opened is a disconnected state', async () => {
		const explorer = new ResultExplorer(() => { });
		explorer.addSession('ggsql-abc', {
			callMethod: async () => { throw { message: 'This query result is no longer available.' }; },
		});
		const r = await explorer.handleRpc({ method: 'get_state', uri: ID, params: {} });
		const s = r.result as Record<string, unknown>;
		assert.strictEqual(s.connected, false);
		assert.strictEqual(s.error_message, 'This query result is no longer available.');
		const page = await explorer.handleRpc({ method: 'get_schema', uri: ID, params: { column_indices: [0] } });
		assert.strictEqual(page.error_message, 'This query result is no longer available.');
	});

	test('forgetting a session tells its open explorers to ask for their state again', async () => {
		const events: { uri?: string; method: string }[] = [];
		const explorer = new ResultExplorer(e => events.push(e as { uri?: string; method: string }));
		let open = 0;
		explorer.addSession('ggsql-abc', {
			callMethod: async () => {
				open += 1;
				return { view: `v${open}`, key: '0f9e', title: 'Query [1]', query: '', columns: [], num_rows: 1 };
			},
		});
		await explorer.handleRpc({ method: 'get_state', uri: ID, params: {} });
		explorer.forgetViews('ggsql-abc');
		explorer.forgetViews('ggsql-other');
		assert.deepStrictEqual(events, [{ uri: ID, method: 'schema_update', params: {} }]);
		// The next request opens a fresh view rather than reusing the old one.
		await explorer.handleRpc({ method: 'get_state', uri: ID, params: {} });
		assert.strictEqual(open, 2);
	});

	test('a session that is gone is an error, not a hang', async () => {
		const explorer = new ResultExplorer(() => { });
		const r = await explorer.handleRpc({ method: 'get_schema', uri: ID, params: { column_indices: [0] } });
		assert.match(r.error_message ?? '', /no longer running/);
	});

	test('get_data_values batches columns that ask for the same rows', async () => {
		const { calls, rpc } = setup({
			result_page: p => ({
				columns: (p.columns as number[]).map(c =>
					c === 1 ? ['1.5', 'NaN', null] : c === 2 ? ['NaN', 'inf', null] : ['a']),
			}),
		});
		const r = await rpc('get_data_values', {
			columns: [
				{ column_index: 1, spec: { first_index: 0, last_index: 2 } },
				{ column_index: 0, spec: { indices: [5] } },
				{ column_index: 2, spec: { first_index: 0, last_index: 2 } },
			],
			format_options: {},
		});
		const pages = calls.filter(c => c[0] === 'result_page').map(c => c[1]);
		assert.deepStrictEqual(pages, [
			{ view: 'v1', rows: { first: 0, last: 2 }, columns: [1, 2] },
			{ view: 'v1', rows: { indices: [5] }, columns: [0] },
		]);
		// NaN is special only in a floating column.
		assert.deepStrictEqual(r.result, { columns: [['1.5', 2, 0], ['a'], ['NaN', 'inf', 0]] });
	});

	test('set_row_filters translates, and marks the filters the kernel rejects', async () => {
		const { calls, rpc } = setup({
			result_set_filters: () => ({ num_rows: 7, errors: [null, 'Not a number: "x"'] }),
			result_page: () => ({ columns: [] }),
		});
		const filters = [
			filter('compare', 0, { op: '>', value: '1' }),
			filter('nonsense', 0),
			{ ...filter('between', 1, { left_value: 'x', right_value: '2' }), error_message: 'stale' },
		];
		const r = await rpc('set_row_filters', { filters });
		assert.deepStrictEqual(r.result, { selected_num_rows: 7, had_errors: true });
		assert.deepStrictEqual(calls.at(-1), ['result_set_filters', {
			view: 'v1',
			filters: [
				{ type: 'compare', column: 0, op: '>', value: '1' },
				{ type: 'between', column: 1, low: 'x', high: '2', negate: false },
			],
		}]);

		const state = (await rpc('get_state')).result as Record<string, any>;
		assert.deepStrictEqual(state.table_shape, { num_rows: 7, num_columns: 3 });
		assert.deepStrictEqual(
			state.row_filters.map((f: RowFilter) => [f.filter_id, f.is_valid, f.error_message]),
			[
				['f-compare', true, undefined],
				['f-nonsense', false, 'Unsupported filter type "nonsense"'],
				['f-between', false, 'Not a number: "x"'],
			]
		);
	});

	test('set_sort_columns passes the keys on and keeps them for get_state', async () => {
		const { calls, rpc } = setup({ result_set_sort: () => ({}) });
		const sort_keys = [{ column_index: 2, ascending: false }];
		assert.deepStrictEqual((await rpc('set_sort_columns', { sort_keys })).result, {});
		assert.deepStrictEqual(calls.at(-1), ['result_set_sort', { view: 'v1', keys: [{ column: 2, ascending: false }] }]);
		const state = (await rpc('get_state')).result as Record<string, any>;
		assert.deepStrictEqual(state.sort_keys, sort_keys);
	});

	test('a kernel error is the error_message', async () => {
		const { rpc } = setup({ result_set_sort: () => { throw { message: 'There is no column 9' }; } });
		const r = await rpc('set_sort_columns', { sort_keys: [{ column_index: 9, ascending: true }] });
		assert.strictEqual(r.error_message, 'There is no column 9');
	});

	test('get_column_profiles replies at once, then sends the profiles', async () => {
		const { calls, events, rpc } = setup({
			result_profile: p => (p.column === 0
				? {
					null_count: 2,
					summary: { min: '1', max: '9', mean: '5', median: '5', stdev: '2' },
					histogram: { bin_edges: ['1', '9'], bin_counts: [98], quantiles: [{ q: 0.5, value: '5' }] },
				}
				: { frequency: { values: ['a'], counts: [3], other_count: 4 } }),
		});
		const r = await rpc('get_column_profiles', {
			callback_id: 'cb',
			profiles: [
				{
					column_index: 0, profiles: [
						{ profile_type: 'null_count' },
						{ profile_type: 'summary_stats' },
						{ profile_type: 'small_histogram', params: { method: 'fixed', num_bins: 10, quantiles: [0.5] } },
					],
				},
				{ column_index: 2, profiles: [{ profile_type: 'small_frequency_table', params: { limit: 5 } }] },
			],
			format_options: {},
		});
		assert.deepStrictEqual(r.result, {});
		await new Promise(resolve => setTimeout(resolve, 0));

		assert.deepStrictEqual(calls.filter(c => c[0] === 'result_profile').map(c => c[1]), [
			{ view: 'v1', column: 0, profiles: { null_count: true, summary: true, histogram: { bins: 10, quantiles: [0.5] } } },
			{ view: 'v1', column: 2, profiles: { frequency: { limit: 5 } } },
		]);
		assert.deepStrictEqual(events, [{
			uri: ID,
			method: 'return_column_profiles',
			params: {
				callback_id: 'cb',
				profiles: [
					{
						null_count: 2,
						summary_stats: {
							type_display: 'integer',
							number_stats: { min_value: '1', max_value: '9', mean: '5', median: '5', stdev: '2' },
						},
						small_histogram: { bin_edges: ['1', '9'], bin_counts: [98], quantiles: [{ q: 0.5, value: '5', exact: true }] },
					},
					{ small_frequency_table: { values: ['a'], counts: [3], other_count: 4 } },
				],
			},
		}]);
	});

	test('export_data_selection covers every selection kind', async () => {
		const { calls, rpc } = setup({
			result_export: () => ({ data: 'csv' }),
			result_page: () => ({ columns: [['42']] }),
		});
		const cases: Array<[object, object | undefined, number[]]> = [
			[{ kind: 'cell_range', selection: { first_row_index: 1, last_row_index: 2, first_column_index: 0, last_column_index: 1 } }, { first: 1, last: 2 }, [0, 1]],
			[{ kind: 'cell_indices', selection: { row_indices: [4, 1], column_indices: [2] } }, { indices: [4, 1] }, [2]],
			[{ kind: 'column_range', selection: { first_index: 1, last_index: 2 } }, undefined, [1, 2]],
			[{ kind: 'row_range', selection: { first_index: 3, last_index: 4 } }, { first: 3, last: 4 }, [0, 1, 2]],
			[{ kind: 'column_indices', selection: { indices: [2, 0] } }, undefined, [2, 0]],
			[{ kind: 'row_indices', selection: { indices: [7] } }, { indices: [7] }, [0, 1, 2]],
		];
		for (const [selection, rows, columns] of cases) {
			const r = await rpc('export_data_selection', { selection, format: 'tsv' });
			assert.deepStrictEqual(r.result, { data: 'csv', format: 'tsv' });
			const expected: Record<string, unknown> = { view: 'v1', columns, format: 'tsv' };
			if (rows) {
				expected.rows = rows;
			}
			assert.deepStrictEqual(calls.at(-1), ['result_export', expected], JSON.stringify(selection));
		}

		const cell = await rpc('export_data_selection', {
			selection: { kind: 'single_cell', selection: { row_index: 3, column_index: 1 } },
			format: 'csv',
		});
		assert.deepStrictEqual(cell.result, { data: '42', format: 'csv' });
		assert.deepStrictEqual(calls.at(-1), ['result_page', { view: 'v1', rows: { indices: [3] }, columns: [1] }]);
	});

	test('convert_to_code is the kernel SQL', async () => {
		const { rpc } = setup({ result_sql: () => ({ sql: ['SELECT *', 'FROM t'] }) });
		const r = await rpc('convert_to_code', { code_syntax_name: { code_syntax_name: 'SQL' } });
		assert.deepStrictEqual(r.result, { converted_code: ['SELECT *', 'FROM t'] });
		assert.deepStrictEqual((await rpc('suggest_code_syntax')).result, { code_syntax_name: 'SQL' });
	});

	test('the unsupported requests answer emptily, and unknown ones are errors', async () => {
		const { rpc } = setup();
		assert.deepStrictEqual((await rpc('get_row_labels', {})).result, { row_labels: [] });
		assert.strictEqual((await rpc('set_column_filters', { filters: [] })).result, null);
		assert.match((await rpc('nope')).error_message ?? '', /Unknown method/);
	});

	test('closing a dataset closes its view, and the next request reopens', async () => {
		const { explorer, calls, rpc } = setup({ result_close: () => ({}) });
		await rpc('get_state');
		explorer.closeDataset(ID);
		await new Promise(resolve => setTimeout(resolve, 0));
		assert.deepStrictEqual(calls.at(-1), ['result_close', { view: 'v1' }]);
		await rpc('get_state');
		assert.strictEqual(calls.filter(c => c[0] === 'result_open').length, 2);
	});
});

suite('Data Explorer translation', () => {
	test('every row filter type maps to a kernel filter', () => {
		const cases: Array<[RowFilter, unknown]> = [
			[filter('not_between', 0, { left_value: '1', right_value: '2' }), { type: 'between', column: 0, low: '1', high: '2', negate: true }],
			[filter('is_null', 1), { type: 'is_null', column: 1, negate: false }],
			[filter('not_null', 1), { type: 'is_null', column: 1, negate: true }],
			[filter('is_empty', 2), { type: 'is_empty', column: 2, negate: false }],
			[filter('not_empty', 2), { type: 'is_empty', column: 2, negate: true }],
			[filter('is_true', 0), { type: 'is_true', column: 0 }],
			[filter('is_false', 0), { type: 'is_false', column: 0 }],
			[filter('search', 2, { search_type: 'regex_match', term: '^a', case_sensitive: true }), { type: 'text', column: 2, match: 'regex', term: '^a', case_sensitive: true }],
			[filter('search', 2, { search_type: 'not_contains', term: 'b', case_sensitive: false }), { type: 'text', column: 2, match: 'not_contains', term: 'b', case_sensitive: false }],
			[filter('set_membership', 2, { values: ['a'], inclusive: true }), { type: 'in', column: 2, values: ['a'], negate: false }],
			[filter('set_membership', 2, { values: ['a'], inclusive: false }), { type: 'in', column: 2, values: ['a'], negate: true }],
			[filter('compare', 0, { op: '~', value: '1' }), { error: 'Unknown comparison "~"' }],
			[filter('compare', 0, { op: '=' }), { error: 'The filter is missing a value' }],
		];
		for (const [input, expected] of cases) {
			assert.deepStrictEqual(toKernelFilter(input), expected, input.filter_type);
		}
	});

	test('search_schema matches names and types, and sorts', () => {
		const search = (filters: object[], sort_order = 'original') =>
			searchSchema({ filters, sort_order } as never, COLUMNS);
		const text = (search_type: string, term: string, case_sensitive = false) =>
			({ filter_type: 'text_search', params: { search_type, term, case_sensitive } });

		assert.deepStrictEqual(search([]), [0, 1, 2]);
		assert.deepStrictEqual(search([text('contains', 'l')]), [2]);
		assert.deepStrictEqual(search([text('contains', 'B', true)]), []);
		assert.deepStrictEqual(search([text('not_contains', 'n')]), [1, 2]);
		assert.deepStrictEqual(search([text('starts_with', 'la')]), [2]);
		assert.deepStrictEqual(search([text('ends_with', 'X')]), [1]);
		assert.deepStrictEqual(search([text('regex_match', '^[nx]$')]), [0, 1]);
		assert.deepStrictEqual(search([text('regex_match', '(')]), []);
		assert.deepStrictEqual(
			search([{ filter_type: 'match_data_types', params: { display_types: ['string', 'floating'] } }]),
			[1, 2]
		);
		assert.deepStrictEqual(search([], 'descending_name'), [1, 0, 2]);
		assert.deepStrictEqual(search([], 'ascending_type'), [1, 0, 2]);
	});

	test('large profiles get a kernel call of their own', () => {
		const calls = toProfileCalls([
			{ profile_type: 'small_histogram', params: { method: 'fixed', num_bins: 10 } },
			{ profile_type: 'large_histogram', params: { method: 'fixed', num_bins: 100 } },
			{ profile_type: 'large_frequency_table', params: { limit: 50 } },
		]);
		assert.deepStrictEqual(calls, [
			{ request: { histogram: { bins: 10, quantiles: [] } }, histogramKey: 'small_histogram' },
			{
				request: { histogram: { bins: 100, quantiles: [] }, frequency: { limit: 50 } },
				histogramKey: 'large_histogram',
				frequencyKey: 'large_frequency_table',
			},
		]);
	});

	test('summaries take the shape of their column kind', () => {
		assert.deepStrictEqual(toSummaryStats('boolean', { true_count: 3, false_count: 1 }), {
			type_display: 'boolean', boolean_stats: { true_count: 3, false_count: 1 },
		});
		assert.deepStrictEqual(toSummaryStats('string', { num_unique: 3, num_empty: 0 }), {
			type_display: 'string', string_stats: { num_unique: 3, num_empty: 0 },
		});
		assert.deepStrictEqual(toSummaryStats('date', { min: '2024-01-01', max: '2024-02-01', num_unique: 2 }), {
			type_display: 'date', date_stats: { min_date: '2024-01-01', max_date: '2024-02-01', num_unique: 2 },
		});
		assert.deepStrictEqual(toSummaryStats('datetime', {}).datetime_stats, {
			min_date: undefined, max_date: undefined, num_unique: undefined,
		});
		assert.deepStrictEqual(toSummaryStats('time', {}), { type_display: 'time' });
	});

	test('cells map nulls and non-finite floats to their codes', () => {
		assert.strictEqual(cellValue(null, 'string'), 0);
		assert.strictEqual(cellValue('NaN', 'floating'), 2);
		assert.strictEqual(cellValue('inf', 'floating'), 10);
		assert.strictEqual(cellValue('-inf', 'floating'), 11);
		assert.strictEqual(cellValue('-inf', 'string'), '-inf');
		assert.strictEqual(cellValue('2.5', 'floating'), '2.5');
	});
});
