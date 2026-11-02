import * as assert from 'assert';
import { ggsqlDataImporter } from '../dataImporter';
import type * as positron from '@posit-dev/positron';
import type * as vscode from 'vscode';

function request(
    filePath: string,
    overrides: Partial<positron.DataImportRequest> = {},
): positron.DataImportRequest {
    return {
        fileUri: { fsPath: filePath } as vscode.Uri,
        variableName: 'my_table',
        options: {},
        ...overrides,
    };
}

/** Generate code synchronously, failing the test if the importer defers or declines. */
function gen(req: positron.DataImportRequest): positron.DataImportResult {
    const result = ggsqlDataImporter.generateCode(req);
    assert.ok(result !== undefined && result !== null && !(result instanceof Promise));
    return result as positron.DataImportResult;
}

/**
 * Assert that `unsupported` contains only the standing DuckDB-backend note,
 * i.e. nothing the user explicitly chose was flagged as unsupported.
 */
function assertNoUnsupported(result: positron.DataImportResult): void {
    assert.deepStrictEqual(
        result.unsupported,
        ['Generated code requires a ggsql session using the default DuckDB backend'],
    );
}

suite('ggsqlDataImporter.generateCode', () => {
    test('generates a plain import for a csv file', () => {
        const result = gen(request('/data/penguins.csv'));
        assert.strictEqual(
            result?.code,
            'CREATE TABLE "my_table" AS\nSELECT *\nFROM \'/data/penguins.csv\';',
        );
        assertNoUnsupported(result);
    });

    test('escapes single quotes in file paths', () => {
        const result = gen(request("/data/bob's file.csv"));
        assert.ok(result?.code.includes("'/data/bob''s file.csv'"));
    });

    test('uses read_csv with header = false when hasHeaderRow is false', () => {
        const result = gen(
            request('/data/p.csv', { options: { hasHeaderRow: false } }),
        );
        assert.ok(result?.code.includes("read_csv('/data/p.csv', header = false)"));
        assertNoUnsupported(result);
    });

    test('treats an extensionless file in a dotted directory as unknown type', () => {
        const result = gen(
            request('/data.v2/penguins', { options: { hasHeaderRow: false } }),
        );
        assert.ok(result?.code.includes("FROM '/data.v2/penguins'"));
        assert.ok(!result?.code.includes('read_csv'));
        assert.ok(result?.unsupported?.some((u) => u.includes('Header row')));
    });

    test('flags header option as unsupported for non-csv files', () => {
        const result = gen(
            request('/data/p.parquet', { options: { hasHeaderRow: false } }),
        );
        assert.ok(result?.unsupported?.some((u) => u.includes('Header row')));
    });

    test('flags worksheet selection as unsupported', () => {
        const result = gen(
            request('/data/p.csv', { options: { sheetName: 'Sheet2' } }),
        );
        assert.ok(result?.unsupported?.some((u) => u.includes('Sheet2')));
    });

    test('translates filters and sorts from the view', () => {
        const result = gen(
            request('/data/p.csv', {
                view: {
                    rowFilters: [
                        {
                            filterType: 'compare',
                            columnName: 'bill_length_mm',
                            columnType: 'number',
                            condition: 'and',
                            op: '>',
                            value: '40',
                        },
                        {
                            filterType: 'set_membership',
                            columnName: 'species',
                            columnType: 'string',
                            condition: 'and',
                            values: ['Adelie', 'Chinstrap'],
                            inclusive: true,
                        },
                        {
                            filterType: 'search',
                            columnName: 'island',
                            columnType: 'string',
                            condition: 'or',
                            searchType: 'contains',
                            term: 'Tor',
                            caseSensitive: false,
                        },
                    ],
                    sortKeys: [
                        { columnName: 'bill_length_mm', ascending: false },
                    ],
                },
            }),
        );
        assert.strictEqual(
            result?.code,
            'CREATE TABLE "my_table" AS\n' +
                'SELECT *\n' +
                "FROM '/data/p.csv'\n" +
                'WHERE "bill_length_mm" > 40\n' +
                '  AND "species" IN (\'Adelie\', \'Chinstrap\')\n' +
                '  OR "island" ILIKE \'%Tor%\' ESCAPE \'\\\\\'\n' +
                'ORDER BY "bill_length_mm" DESC;',
        );
        assertNoUnsupported(result);
    });

    test('escapes LIKE wildcards in search terms', () => {
        const result = gen(
            request('/data/p.csv', {
                view: {
                    rowFilters: [
                        { filterType: 'search', columnName: 'name', columnType: 'string', condition: 'and', searchType: 'contains', term: '50%_off', caseSensitive: true },
                    ],
                    sortKeys: [],
                },
            }),
        );
        assert.ok(result?.code.includes(`LIKE '%50\\%\\_off%' ESCAPE '\\\\'`));
    });

    test('renders mixed conditions flatly when OR precedes AND', () => {
        // Mirrors Data Explorer's left-to-right semantics: SQL parses this as
        // `A OR (B AND C)` due to precedence, which is the intended behaviour.
        const result = gen(
            request('/data/p.csv', {
                view: {
                    rowFilters: [
                        { filterType: 'compare', columnName: 'a', columnType: 'integer', condition: 'and', op: '=', value: '1' },
                        { filterType: 'compare', columnName: 'b', columnType: 'integer', condition: 'or', op: '=', value: '2' },
                        { filterType: 'compare', columnName: 'c', columnType: 'integer', condition: 'and', op: '=', value: '3' },
                    ],
                    sortKeys: [],
                },
            }),
        );
        assert.ok(result?.code.includes(
            'WHERE "a" = 1\n  OR "b" = 2\n  AND "c" = 3;',
        ));
    });

    test('translates remaining filter kinds', () => {
        const result = gen(
            request('/data/p.csv', {
                view: {
                    rowFilters: [
                        { filterType: 'between', columnName: 'x', columnType: 'integer', condition: 'and', leftValue: '1', rightValue: '5' },
                        { filterType: 'not_between', columnName: 'y', columnType: 'integer', condition: 'and', leftValue: '0', rightValue: '10' },
                        { filterType: 'is_null', columnName: 'z', columnType: 'string', condition: 'and' },
                        { filterType: 'not_null', columnName: 'w', columnType: 'string', condition: 'and' },
                        { filterType: 'is_empty', columnName: 'v', columnType: 'string', condition: 'and' },
                        { filterType: 'is_true', columnName: 'b', columnType: 'boolean', condition: 'and' },
                    ],
                    sortKeys: [],
                },
            }),
        );
        assert.strictEqual(
            result?.code,
            'CREATE TABLE "my_table" AS\n' +
                'SELECT *\n' +
                "FROM '/data/p.csv'\n" +
                'WHERE "x" BETWEEN 1 AND 5\n' +
                '  AND "y" NOT BETWEEN 0 AND 10\n' +
                '  AND "z" IS NULL\n' +
                '  AND "w" IS NOT NULL\n' +
                '  AND "v" = \'\'\n' +
                '  AND "b";',
        );
    });

    test('renders regex search with case-insensitive flag when needed', () => {
        const result = gen(
            request('/data/p.csv', {
                view: {
                    rowFilters: [
                        { filterType: 'search', columnName: 'name', columnType: 'string', condition: 'and', searchType: 'regex_match', term: '^a+$', caseSensitive: false },
                    ],
                    sortKeys: [],
                },
            }),
        );
        assert.ok(result?.code.includes(`regexp_matches("name", '^a+$', 'i')`));
    });
});
