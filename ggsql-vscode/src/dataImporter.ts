/*
 * ggsql data importer.
 *
 * Registers a Data Explorer importer so that dragging a csv/parquet/json file
 * into Positron offers "ggsql" as a way to load it. The generated code is a
 * ggsql query that reads the file into a table, reproducing any row filters
 * and sorts from the current Data Explorer view.
 */

import * as positron from '@posit-dev/positron';

/** File extensions ggsql can read through its duckdb backend. */
const READABLE_EXTENSIONS = ['csv', 'tsv', 'parquet', 'json', 'jsonl', 'ndjson'];

/** Column types whose stringified values can be embedded in SQL unquoted. */
const NUMERIC_TYPES = new Set(['integer', 'number', 'float', 'double', 'decimal']);

/** A small list of SQL reserved words, so Positron can suffix colliding variable names. */
const RESERVED_NAMES = [
    'select', 'from', 'where', 'table', 'group', 'order', 'by', 'insert',
    'update', 'delete', 'create', 'drop', 'join', 'union', 'all', 'and',
    'or', 'not', 'null', 'as', 'on', 'in', 'between', 'like', 'limit',
];

/** Quote a string value for SQL, escaping embedded quotes by doubling. */
function quoteString(value: string): string {
    return `'${value.replace(/'/g, "''")}'`;
}

/** Quote an identifier (column or table name) for SQL. */
function quoteIdentifier(name: string): string {
    return `"${name.replace(/"/g, '""')}"`;
}

/**
 * Render a stringified filter value as a SQL literal, using the column's
 * display type to decide whether quoting is needed.
 */
function renderValue(value: string, columnType: string): string {
    const type = columnType.toLowerCase();
    if (NUMERIC_TYPES.has(type)) {
        return value;
    }
    if (type === 'boolean') {
        return value.toLowerCase() === 'true' ? 'TRUE' : 'FALSE';
    }
    return quoteString(value);
}

/** Translate one Data Explorer row filter into a SQL predicate. */
function renderFilter(filter: positron.DataImportRowFilter): string | undefined {
    const column = quoteIdentifier(filter.columnName);
    switch (filter.filterType) {
        case 'between':
            return `${column} BETWEEN ${renderValue(filter.leftValue, filter.columnType)} AND ${renderValue(filter.rightValue, filter.columnType)}`;
        case 'not_between':
            return `${column} NOT BETWEEN ${renderValue(filter.leftValue, filter.columnType)} AND ${renderValue(filter.rightValue, filter.columnType)}`;
        case 'compare':
            return `${column} ${filter.op} ${renderValue(filter.value, filter.columnType)}`;
        case 'search': {
            const like = filter.caseSensitive ? 'LIKE' : 'ILIKE';
            switch (filter.searchType) {
                case 'contains':
                    return `${column} ${like} ${quoteString(`%${filter.term}%`)}`;
                case 'not_contains':
                    return `${column} NOT ${like} ${quoteString(`%${filter.term}%`)}`;
                case 'starts_with':
                    return `${column} ${like} ${quoteString(`${filter.term}%`)}`;
                case 'ends_with':
                    return `${column} ${like} ${quoteString(`%${filter.term}`)}`;
                case 'regex_match':
                    return filter.caseSensitive
                        ? `regexp_matches(${column}, ${quoteString(filter.term)})`
                        : `regexp_matches(${column}, ${quoteString(filter.term)}, 'i')`;
            }
            break;
        }
        case 'set_membership': {
            const values = filter.values.map((v) => renderValue(v, filter.columnType)).join(', ');
            return `${column} ${filter.inclusive ? 'IN' : 'NOT IN'} (${values})`;
        }
        case 'is_null':
            return `${column} IS NULL`;
        case 'not_null':
            return `${column} IS NOT NULL`;
        case 'is_empty':
            return `${column} = ''`;
        case 'not_empty':
            return `${column} <> ''`;
        case 'is_true':
            return `${column}`;
        case 'is_false':
            return `NOT ${column}`;
    }
    return undefined;
}

/** Build the FROM clause, honouring import options that need explicit reader calls. */
function renderFrom(filePath: string, extension: string, options: positron.DataImportOptions): string {
    if (
        options.hasHeaderRow === false &&
        (extension === 'csv' || extension === 'tsv')
    ) {
        return `read_csv(${quoteString(filePath)}, header = false)`;
    }
    return quoteString(filePath);
}

/** The ggsql data importer offered by the Data Explorer import dialog. */
export const ggsqlDataImporter: positron.DataImporter = {
    languageId: 'ggsql',
    displayName: 'ggsql',
    fileExtensions: READABLE_EXTENSIONS,
    reservedNames: RESERVED_NAMES,

    generateCode(request: positron.DataImportRequest): positron.DataImportResult {
        const filePath = request.fileUri.fsPath;
        const extension = filePath.split('.').pop()?.toLowerCase() ?? '';
        const unsupported: string[] = [];

        if (request.options.sheetName !== undefined) {
            unsupported.push(`Worksheet selection ('${request.options.sheetName}')`);
        }
        if (
            request.options.hasHeaderRow === false &&
            extension !== 'csv' &&
            extension !== 'tsv'
        ) {
            unsupported.push('Header row option (only supported for csv/tsv files)');
        }

        const lines = [
            `CREATE TABLE ${quoteIdentifier(request.variableName)} AS`,
            'SELECT *',
            `FROM ${renderFrom(filePath, extension, request.options)}`,
        ];

        const view = request.view;
        if (view && view.rowFilters.length > 0) {
            const predicates: string[] = [];
            for (const filter of view.rowFilters) {
                const predicate = renderFilter(filter);
                if (predicate === undefined) {
                    unsupported.push(`Row filter on ${filter.columnName} (${filter.filterType})`);
                    continue;
                }
                predicates.push(predicates.length === 0 ? predicate : `${filter.condition.toUpperCase()} ${predicate}`);
            }
            if (predicates.length > 0) {
                lines.push(`WHERE ${predicates.join('\n  ')}`);
            }
        }

        if (view && view.sortKeys.length > 0) {
            const keys = view.sortKeys.map(
                (key) => `${quoteIdentifier(key.columnName)} ${key.ascending ? 'ASC' : 'DESC'}`
            );
            lines.push(`ORDER BY ${keys.join(', ')}`);
        }

        return {
            code: lines.join('\n') + ';',
            unsupported: unsupported.length > 0 ? unsupported : undefined,
        };
    },
};
