/*
 * The two protocols the Data Explorer backend translates between.
 *
 * Positron's half is a hand-written subset of its Data Explorer backend
 * protocol (`positronDataExplorerComm.ts` in Positron): only the requests,
 * results and events this backend answers, with the fields it reads or
 * writes. The kernel's half is the frontend-neutral results API that
 * `ggsql-jupyter/src/results/methods.rs` answers.
 */

// --- Positron's Data Explorer protocol ---

/** A column's display type. Every kernel `kind` is one of these. */
export type ColumnDisplayType =
    | 'boolean' | 'string' | 'date' | 'datetime' | 'time' | 'interval' | 'object'
    | 'array' | 'struct' | 'unknown' | 'floating' | 'integer' | 'decimal';

export type SupportStatus = 'supported' | 'unsupported';

export interface TableShape {
    num_rows: number;
    num_columns: number;
}

export interface ColumnSchema {
    column_name: string;
    column_index: number;
    type_name: string;
    type_display: ColumnDisplayType;
}

export type RowFilterType =
    | 'between' | 'compare' | 'is_empty' | 'is_false' | 'is_null' | 'is_true'
    | 'not_between' | 'not_empty' | 'not_null' | 'search' | 'set_membership';

export type TextSearchType = 'contains' | 'not_contains' | 'starts_with' | 'ends_with' | 'regex_match';

export interface FilterBetween {
    left_value: string;
    right_value: string;
}

export interface FilterComparison {
    op: '=' | '!=' | '<' | '<=' | '>' | '>=';
    value: string;
}

export interface FilterSetMembership {
    values: string[];
    inclusive: boolean;
}

export interface FilterTextSearch {
    search_type: TextSearchType;
    term: string;
    case_sensitive: boolean;
}

export interface FilterMatchDataTypes {
    display_types: ColumnDisplayType[];
}

export interface RowFilter {
    filter_id: string;
    filter_type: RowFilterType;
    column_schema: ColumnSchema;
    condition: 'and' | 'or';
    is_valid?: boolean;
    error_message?: string;
    params?: FilterBetween | FilterComparison | FilterTextSearch | FilterSetMembership;
}

export interface ColumnFilter {
    filter_type: 'text_search' | 'match_data_types';
    params: FilterTextSearch | FilterMatchDataTypes;
}

export interface ColumnSortKey {
    column_index: number;
    ascending: boolean;
}

export type SearchSchemaSortOrder =
    | 'original' | 'ascending_name' | 'descending_name' | 'ascending_type' | 'descending_type';

export interface SearchSchemaParams {
    filters: ColumnFilter[];
    sort_order: SearchSchemaSortOrder;
}

export interface DataSelectionSingleCell {
    row_index: number;
    column_index: number;
}

export interface DataSelectionCellRange {
    first_row_index: number;
    last_row_index: number;
    first_column_index: number;
    last_column_index: number;
}

export interface DataSelectionCellIndices {
    row_indices: number[];
    column_indices: number[];
}

export interface DataSelectionRange {
    first_index: number;
    last_index: number;
}

export interface DataSelectionIndices {
    indices: number[];
}

export type ArraySelection = DataSelectionRange | DataSelectionIndices;

export interface ColumnSelection {
    column_index: number;
    spec: ArraySelection;
}

export type TableSelection =
    | { kind: 'single_cell'; selection: DataSelectionSingleCell }
    | { kind: 'cell_range'; selection: DataSelectionCellRange }
    | { kind: 'cell_indices'; selection: DataSelectionCellIndices }
    | { kind: 'column_range' | 'row_range'; selection: DataSelectionRange }
    | { kind: 'column_indices' | 'row_indices'; selection: DataSelectionIndices };

export type ExportFormat = 'csv' | 'tsv' | 'html';

/** A cell: formatted text, or a code for a special value (see `format.ts`). */
export type ColumnValue = number | string;

export type ColumnProfileType =
    | 'null_count' | 'summary_stats' | 'small_frequency_table' | 'large_frequency_table'
    | 'small_histogram' | 'large_histogram';

export interface ColumnHistogramParams {
    method: string;
    num_bins: number;
    quantiles?: number[];
}

export interface ColumnFrequencyTableParams {
    limit: number;
}

export interface ColumnProfileSpec {
    profile_type: ColumnProfileType;
    params?: ColumnHistogramParams | ColumnFrequencyTableParams;
}

export interface ColumnProfileRequest {
    column_index: number;
    profiles: ColumnProfileSpec[];
}

export interface GetColumnProfilesParams {
    callback_id: string;
    profiles: ColumnProfileRequest[];
}

export interface ColumnHistogram {
    bin_edges: string[];
    bin_counts: number[];
    quantiles: { q: number; value: string; exact: boolean }[];
}

export interface ColumnFrequencyTable {
    values: ColumnValue[];
    counts: number[];
    other_count?: number;
}

export interface ColumnSummaryStats {
    type_display: ColumnDisplayType;
    number_stats?: { min_value?: string; max_value?: string; mean?: string; median?: string; stdev?: string };
    string_stats?: { num_empty: number; num_unique: number };
    boolean_stats?: { true_count: number; false_count: number };
    date_stats?: { num_unique?: number; min_date?: string; max_date?: string };
    datetime_stats?: { num_unique?: number; min_date?: string; max_date?: string };
}

export interface ColumnProfileResult {
    null_count?: number;
    summary_stats?: ColumnSummaryStats;
    small_histogram?: ColumnHistogram;
    large_histogram?: ColumnHistogram;
    small_frequency_table?: ColumnFrequencyTable;
    large_frequency_table?: ColumnFrequencyTable;
}

export interface BackendState {
    display_name: string;
    table_shape: TableShape;
    table_unfiltered_shape: TableShape;
    has_row_labels: boolean;
    column_filters: ColumnFilter[];
    row_filters: RowFilter[];
    sort_keys: ColumnSortKey[];
    supported_features: unknown;
    connected?: boolean;
    error_message?: string;
}

// --- The kernel's results API ---

export type ColumnKind =
    | 'integer' | 'floating' | 'boolean' | 'string' | 'date' | 'datetime' | 'time' | 'unknown';

export interface KernelColumn {
    name: string;
    type_name: string;
    kind: ColumnKind;
}

/** `result_open`'s result. */
export interface OpenedResult {
    view: string;
    key: string;
    title: string;
    query: string;
    columns: KernelColumn[];
    num_rows: number;
}

/** Rows of a view: `first..=last`, or these positions in this order. */
export type KernelRows = { first: number; last: number } | { indices: number[] };

export type KernelFilter =
    | { type: 'compare'; column: number; op: FilterComparison['op']; value: string }
    | { type: 'between'; column: number; low: string; high: string; negate: boolean }
    | { type: 'is_null'; column: number; negate: boolean }
    | { type: 'is_empty'; column: number; negate: boolean }
    | { type: 'is_true'; column: number }
    | { type: 'is_false'; column: number }
    | {
        type: 'text';
        column: number;
        match: 'contains' | 'not_contains' | 'starts_with' | 'ends_with' | 'regex';
        term: string;
        case_sensitive: boolean;
    }
    | { type: 'in'; column: number; values: string[]; negate: boolean };

export interface KernelSortKey {
    column: number;
    ascending: boolean;
}

/** `result_profile`'s `profiles` parameter. */
export interface KernelProfileRequest {
    null_count?: boolean;
    summary?: boolean;
    histogram?: { bins: number; quantiles: number[] };
    frequency?: { limit: number };
}

/** `result_profile`'s result. `summary`'s fields depend on the column's kind. */
export interface KernelProfile {
    null_count?: number;
    summary?: Record<string, string | number | undefined>;
    histogram?: { bin_edges: string[]; bin_counts: number[]; quantiles: { q: number; value: string }[] };
    frequency?: { values: string[]; counts: number[]; other_count: number };
}
