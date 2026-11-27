/*
 * Positron's row filters, sort keys and column searches, in the kernel's terms.
 */

import type {
    ColumnFilter,
    ColumnSortKey,
    FilterBetween,
    FilterComparison,
    FilterMatchDataTypes,
    FilterSetMembership,
    FilterTextSearch,
    KernelColumn,
    KernelFilter,
    KernelSortKey,
    RowFilter,
    SearchSchemaParams,
} from './protocol';

const TEXT_MATCH = {
    contains: 'contains',
    not_contains: 'not_contains',
    starts_with: 'starts_with',
    ends_with: 'ends_with',
    regex_match: 'regex',
} as const;

const COMPARE_OPS = new Set(['=', '!=', '<', '<=', '>', '>=']);

/**
 * A row filter as the kernel's filter, or the reason it cannot be one.
 * Values are checked against the column's type by the kernel, not here.
 */
export function toKernelFilter(filter: RowFilter): KernelFilter | { error: string } {
    const column = filter.column_schema?.column_index;
    if (typeof column !== 'number') {
        return { error: 'The filter names no column' };
    }
    const params = filter.params ?? {};
    const missing = { error: 'The filter is missing a value' };

    switch (filter.filter_type) {
        case 'compare': {
            const { op, value } = params as FilterComparison;
            if (!COMPARE_OPS.has(op)) {
                return { error: `Unknown comparison ${JSON.stringify(op)}` };
            }
            if (typeof value !== 'string') {
                return missing;
            }
            return { type: 'compare', column, op, value };
        }
        case 'between':
        case 'not_between': {
            const { left_value: low, right_value: high } = params as FilterBetween;
            if (typeof low !== 'string' || typeof high !== 'string') {
                return missing;
            }
            return { type: 'between', column, low, high, negate: filter.filter_type === 'not_between' };
        }
        case 'is_null':
        case 'not_null':
            return { type: 'is_null', column, negate: filter.filter_type === 'not_null' };
        case 'is_empty':
        case 'not_empty':
            return { type: 'is_empty', column, negate: filter.filter_type === 'not_empty' };
        case 'is_true':
            return { type: 'is_true', column };
        case 'is_false':
            return { type: 'is_false', column };
        case 'search': {
            const { search_type, term, case_sensitive } = params as FilterTextSearch;
            const match = TEXT_MATCH[search_type];
            if (!match) {
                return { error: `Unknown search ${JSON.stringify(search_type)}` };
            }
            if (typeof term !== 'string') {
                return missing;
            }
            return { type: 'text', column, match, term, case_sensitive: !!case_sensitive };
        }
        case 'set_membership': {
            const { values, inclusive } = params as FilterSetMembership;
            if (!Array.isArray(values)) {
                return missing;
            }
            return { type: 'in', column, values, negate: inclusive === false };
        }
        default:
            return { error: `Unsupported filter type ${JSON.stringify(filter.filter_type)}` };
    }
}

export function toKernelSortKeys(keys: ColumnSortKey[]): KernelSortKey[] {
    return keys.map(k => ({ column: k.column_index, ascending: k.ascending }));
}

/** Whether a column passes one of `search_schema`'s filters. */
function columnMatches(filter: ColumnFilter, column: KernelColumn): boolean {
    if (filter.filter_type === 'match_data_types') {
        const { display_types } = filter.params as FilterMatchDataTypes;
        return Array.isArray(display_types) && display_types.includes(column.kind);
    }
    if (filter.filter_type !== 'text_search') {
        return true;
    }
    const { search_type, case_sensitive } = filter.params as FilterTextSearch;
    const term = (filter.params as FilterTextSearch).term ?? '';
    if (search_type === 'regex_match') {
        try {
            return new RegExp(term, case_sensitive ? '' : 'i').test(column.name);
        } catch {
            return false;
        }
    }
    const name = case_sensitive ? column.name : column.name.toLowerCase();
    const needle = case_sensitive ? term : term.toLowerCase();
    switch (search_type ?? 'contains') {
        case 'contains':
            return name.includes(needle);
        case 'not_contains':
            return !name.includes(needle);
        case 'starts_with':
            return name.startsWith(needle);
        case 'ends_with':
            return name.endsWith(needle);
        default:
            return true;
    }
}

function compare(a: string, b: string): number {
    return a < b ? -1 : a > b ? 1 : 0;
}

/** The indices of the columns passing every filter, in the order asked for. */
export function searchSchema(params: SearchSchemaParams, columns: KernelColumn[]): number[] {
    const filters = params.filters ?? [];
    const matches = columns
        .map((_, i) => i)
        .filter(i => filters.every(f => columnMatches(f, columns[i])));
    const name = (i: number) => columns[i].name.toLowerCase();
    const kind = (i: number) => columns[i].kind;
    switch (params.sort_order) {
        case 'ascending_name':
            return matches.sort((a, b) => compare(name(a), name(b)));
        case 'descending_name':
            return matches.sort((a, b) => compare(name(b), name(a)));
        case 'ascending_type':
            return matches.sort((a, b) => compare(kind(a), kind(b)));
        case 'descending_type':
            return matches.sort((a, b) => compare(kind(b), kind(a)));
        default:
            return matches;
    }
}
