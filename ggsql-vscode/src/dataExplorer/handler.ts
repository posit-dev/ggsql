/*
 * The Data Explorer backend for console query results.
 *
 * Positron sends Data Explorer requests for the `ggsql` provider here, and
 * each is answered by calling the kernel's results API (`result_*` methods,
 * see `ggsql-jupyter/src/results/methods.rs`) through `callMethod` on the
 * session that ran the query. A dataset id is `<session id>/<result key>`, and
 * each open dataset holds one kernel view, opened on first use and closed
 * with the Data Explorer.
 */

import type * as positron from '@posit-dev/positron';
import { cellValue, errorMessage, span } from './format';
import { searchSchema, toKernelFilter, toKernelSortKeys } from './filters';
import { addProfile, toProfileCalls } from './profiles';
import type {
    BackendState,
    ColumnProfileResult,
    ColumnSchema,
    ColumnSelection,
    ColumnSortKey,
    ColumnValue,
    ExportFormat,
    GetColumnProfilesParams,
    KernelFilter,
    KernelProfile,
    KernelRows,
    OpenedResult,
    RowFilter,
    SearchSchemaParams,
    TableSelection,
} from './protocol';

/** The provider id the backend registers under. */
export const PROVIDER_ID = 'ggsql';

/** The part of a runtime session the backend uses. */
export interface ResultsSession {
    callMethod?(method: string, ...args: any[]): Thenable<any>;
}

/** A result's reference, `<sessionId>/<key>`, as the console link names it. */
export function resultRef(sessionId: string, key: string): string {
    return `${sessionId}/${key}`;
}

/**
 * A result reference's session and key, or undefined when it is not one.
 * Only the hex keys the kernel produces are accepted.
 */
export function parseResultRef(ref: string): { sessionId: string; key: string } | undefined {
    const [sessionId, key, ...rest] = ref.split('/');
    if (!sessionId || !key || rest.length > 0 || !/^[0-9a-f]+$/.test(key)) {
        return undefined;
    }
    return { sessionId, key };
}

/**
 * The dataset id for result `key` of session `sessionId`: the reference
 * behind a `ggsql:` scheme. Positron puts the id in the explorer editor's URI
 * and parses it back out, accepting only a UUID or `<scheme>:<rest>`
 * (`PositronDataExplorerUri.parse`), so an id without a scheme opens an
 * empty editor.
 */
export function datasetId(sessionId: string, key: string): string {
    return `${PROVIDER_ID}:${resultRef(sessionId, key)}`;
}

/** A dataset id's session and result key, or undefined when it is not one. */
export function parseDatasetId(id: string): { sessionId: string; key: string } | undefined {
    const prefix = `${PROVIDER_ID}:`;
    return id.startsWith(prefix) ? parseResultRef(id.slice(prefix.length)) : undefined;
}

/** One open dataset: its kernel view and what Positron last set on it. */
export interface Dataset {
    sessionId: string;
    result: OpenedResult;
    selectedRows: number;
    rowFilters: RowFilter[];
    sortKeys: ColumnSortKey[];
}

function supported<T extends string>(field: string, types: T[]): Array<Record<string, string>> {
    return types.map(t => ({ [field]: t, support_status: 'supported' }));
}

const SUPPORTED_FEATURES = {
    search_schema: {
        support_status: 'supported',
        supported_types: supported('column_filter_type', ['text_search', 'match_data_types']),
    },
    set_column_filters: {
        support_status: 'unsupported',
        supported_types: [],
    },
    set_row_filters: {
        support_status: 'supported',
        // The kernel ANDs every filter.
        supports_conditions: 'unsupported',
        supported_types: supported('row_filter_type', [
            'between', 'compare', 'is_empty', 'is_false', 'is_null', 'is_true',
            'not_between', 'not_empty', 'not_null', 'search', 'set_membership',
        ]),
    },
    get_column_profiles: {
        support_status: 'supported',
        supported_types: supported('profile_type', [
            'null_count', 'summary_stats', 'small_histogram', 'small_frequency_table',
        ]),
    },
    set_sort_columns: {
        support_status: 'supported',
    },
    export_data_selection: {
        support_status: 'supported',
        supported_formats: ['csv', 'tsv', 'html'],
    },
    convert_to_code: {
        support_status: 'supported',
        code_syntaxes: [{ code_syntax_name: 'SQL' }],
    },
};

/** The state of a dataset whose result could not be opened. */
function disconnectedState(id: string, message: string): BackendState {
    const empty = { num_rows: 0, num_columns: 0 };
    return {
        display_name: id,
        connected: false,
        error_message: message,
        table_shape: empty,
        table_unfiltered_shape: empty,
        has_row_labels: false,
        column_filters: [],
        row_filters: [],
        sort_keys: [],
        supported_features: {
            search_schema: { support_status: 'unsupported', supported_types: [] },
            set_column_filters: { support_status: 'unsupported', supported_types: [] },
            set_row_filters: { support_status: 'unsupported', supports_conditions: 'unsupported', supported_types: [] },
            get_column_profiles: { support_status: 'unsupported', supported_types: [] },
            set_sort_columns: { support_status: 'unsupported' },
            export_data_selection: { support_status: 'unsupported', supported_formats: [] },
            convert_to_code: { support_status: 'unsupported' },
        },
    };
}

export class ResultExplorer {
    private readonly sessions = new Map<string, ResultsSession>();
    private readonly datasets = new Map<string, Promise<Dataset>>();

    constructor(
        private readonly sendUiEvent: (event: positron.DataExplorerUiEvent) => void,
        private readonly log: (message: string) => void = () => { }
    ) { }

    /** Make a session's results reachable. Dispose to forget the session. */
    addSession(sessionId: string, session: ResultsSession): { dispose(): void } {
        this.sessions.set(sessionId, session);
        return {
            dispose: () => {
                if (this.sessions.get(sessionId) === session) {
                    this.sessions.delete(sessionId);
                }
                this.forgetViews(sessionId);
            },
        };
    }

    /**
     * Forget the views of a session whose kernel has exited or restarted,
     * since they went with it, and tell each open Data Explorer its schema
     * changed. That makes it ask for its state again, which opens the result
     * afresh: the explorer then shows the result, or the kernel's explanation
     * of why it is gone, rather than a sort or filter that silently did nothing.
     */
    forgetViews(sessionId: string): void {
        for (const id of [...this.datasets.keys()]) {
            if (parseDatasetId(id)?.sessionId === sessionId) {
                this.datasets.delete(id);
                this.sendUiEvent({ uri: id, method: 'schema_update', params: {} });
            }
        }
    }

    /** The dataset `id`, opening a kernel view of it the first time. */
    open(id: string): Promise<Dataset> {
        let dataset = this.datasets.get(id);
        if (!dataset) {
            dataset = this.openView(id);
            this.datasets.set(id, dataset);
            // A failed open is not kept, so the next request tries again.
            dataset.catch(() => {
                if (this.datasets.get(id) === dataset) {
                    this.datasets.delete(id);
                }
            });
        }
        return dataset;
    }

    private async openView(id: string): Promise<Dataset> {
        const parsed = parseDatasetId(id);
        if (!parsed) {
            throw new Error(`Not a ggsql result: ${id}`);
        }
        const result = await this.call<OpenedResult>(parsed.sessionId, 'result_open', { key: parsed.key });
        return {
            sessionId: parsed.sessionId,
            result,
            selectedRows: result.num_rows,
            rowFilters: [],
            sortKeys: [],
        };
    }

    /** Release the kernel view of a dataset whose Data Explorer closed. */
    closeDataset(id: string): void {
        const dataset = this.datasets.get(id);
        this.datasets.delete(id);
        dataset
            ?.then(ds => this.call(ds.sessionId, 'result_close', { view: ds.result.view }))
            .catch(err => this.log(`Closing ${id} failed: ${errorMessage(err)}`));
    }

    private async call<T>(sessionId: string, method: string, params: object): Promise<T> {
        const session = this.sessions.get(sessionId);
        if (!session?.callMethod) {
            throw new Error('The ggsql session that ran this query is no longer running.');
        }
        return await session.callMethod(method, params) as T;
    }

    async handleRpc(request: positron.DataExplorerRpcRequest): Promise<positron.DataExplorerRpcResponse> {
        try {
            return { result: await this.dispatch(request) };
        } catch (err) {
            return { error_message: errorMessage(err) };
        }
    }

    private async dispatch(request: positron.DataExplorerRpcRequest): Promise<unknown> {
        const id = request.uri;
        if (id === undefined) {
            throw new Error(`No dataset given for ${request.method}`);
        }
        const params = request.params as any;

        if (request.method === 'get_state') {
            let ds: Dataset;
            try {
                ds = await this.open(id);
            } catch (err) {
                return disconnectedState(id, errorMessage(err));
            }
            return this.getState(ds);
        }

        const ds = await this.open(id);
        switch (request.method) {
            case 'open_dataset':
                return {};
            case 'get_schema':
                return this.getSchema(ds, params.column_indices ?? []);
            case 'search_schema':
                return { matches: searchSchema(params as SearchSchemaParams, ds.result.columns) };
            case 'get_data_values':
                return this.getDataValues(ds, params.columns ?? []);
            case 'get_row_labels':
                return { row_labels: [] };
            case 'set_row_filters':
                return this.setRowFilters(ds, params.filters ?? []);
            case 'set_sort_columns':
                return this.setSortColumns(ds, params.sort_keys ?? []);
            case 'set_column_filters':
                // Unsupported, and advertised as such in `get_state`.
                return null;
            case 'get_column_profiles':
                void this.sendColumnProfiles(id, ds, params as GetColumnProfilesParams);
                return {};
            case 'export_data_selection':
                return this.exportDataSelection(ds, params.selection, params.format ?? 'csv');
            case 'convert_to_code': {
                const { sql } = await this.call<{ sql: string[] }>(ds.sessionId, 'result_sql', { view: ds.result.view });
                return { converted_code: sql };
            }
            case 'suggest_code_syntax':
                return { code_syntax_name: 'SQL' };
            default:
                throw new Error(`Unknown method ${JSON.stringify(request.method)}`);
        }
    }

    private getState(ds: Dataset): BackendState {
        const num_columns = ds.result.columns.length;
        return {
            display_name: ds.result.title,
            table_shape: { num_rows: ds.selectedRows, num_columns },
            table_unfiltered_shape: { num_rows: ds.result.num_rows, num_columns },
            has_row_labels: false,
            column_filters: [],
            row_filters: ds.rowFilters,
            sort_keys: ds.sortKeys,
            supported_features: SUPPORTED_FEATURES,
        };
    }

    private getSchema(ds: Dataset, indices: number[]): { columns: ColumnSchema[] } {
        const columns: ColumnSchema[] = [];
        for (const i of indices) {
            const col = ds.result.columns[i];
            if (col) {
                columns.push({
                    column_name: col.name,
                    column_index: i,
                    type_name: col.type_name,
                    type_display: col.kind,
                });
            }
        }
        return { columns };
    }

    /** Read cells; columns asking for the same rows share one `result_page`. */
    private async getDataValues(ds: Dataset, selections: ColumnSelection[]): Promise<{ columns: ColumnValue[][] }> {
        const groups = new Map<string, { rows: KernelRows; slots: number[]; columns: number[] }>();
        selections.forEach((sel, slot) => {
            const rows: KernelRows = 'indices' in sel.spec
                ? { indices: sel.spec.indices }
                : { first: sel.spec.first_index, last: sel.spec.last_index };
            const key = JSON.stringify(rows);
            let group = groups.get(key);
            if (!group) {
                group = { rows, slots: [], columns: [] };
                groups.set(key, group);
            }
            group.slots.push(slot);
            group.columns.push(sel.column_index);
        });

        const out: ColumnValue[][] = selections.map(() => []);
        await Promise.all([...groups.values()].map(async group => {
            const page = await this.call<{ columns: (string | null)[][] }>(ds.sessionId, 'result_page', {
                view: ds.result.view,
                rows: group.rows,
                columns: group.columns,
            });
            group.slots.forEach((slot, j) => {
                const kind = ds.result.columns[group.columns[j]]?.kind ?? 'unknown';
                out[slot] = (page.columns[j] ?? []).map(cell => cellValue(cell, kind));
            });
        }));
        return { columns: out };
    }

    private async setRowFilters(ds: Dataset, filters: RowFilter[]): Promise<{ selected_num_rows: number; had_errors: boolean }> {
        const translated = filters.map(toKernelFilter);
        const errors: (string | null)[] = translated.map(t => 'error' in t ? t.error : null);
        const sent: KernelFilter[] = [];
        const sentAt: number[] = [];
        translated.forEach((t, i) => {
            if (!('error' in t)) {
                sent.push(t);
                sentAt.push(i);
            }
        });

        const reply = await this.call<{ num_rows: number; errors: (string | null)[] }>(
            ds.sessionId, 'result_set_filters', { view: ds.result.view, filters: sent }
        );
        sentAt.forEach((i, j) => {
            errors[i] = reply.errors?.[j] ?? null;
        });

        ds.rowFilters = filters.map((filter, i) => {
            const { error_message: _, ...rest } = filter;
            const error = errors[i];
            return error === null
                ? { ...rest, is_valid: true }
                : { ...rest, is_valid: false, error_message: error };
        });
        ds.selectedRows = reply.num_rows;
        return { selected_num_rows: reply.num_rows, had_errors: errors.some(e => e !== null) };
    }

    private async setSortColumns(ds: Dataset, keys: ColumnSortKey[]): Promise<object> {
        await this.call(ds.sessionId, 'result_set_sort', {
            view: ds.result.view,
            keys: toKernelSortKeys(keys),
        });
        ds.sortKeys = keys;
        return {};
    }

    /** Answer `get_column_profiles` with a `return_column_profiles` event. */
    private async sendColumnProfiles(id: string, ds: Dataset, params: GetColumnProfilesParams): Promise<void> {
        const profiles = await Promise.all((params.profiles ?? []).map(async request => {
            const out: ColumnProfileResult = {};
            const kind = ds.result.columns[request.column_index]?.kind ?? 'unknown';
            for (const call of toProfileCalls(request.profiles ?? [])) {
                try {
                    const profile = await this.call<KernelProfile>(ds.sessionId, 'result_profile', {
                        view: ds.result.view,
                        column: request.column_index,
                        profiles: call.request,
                    });
                    addProfile(out, call, profile, kind);
                } catch (err) {
                    this.log(`Profiling column ${request.column_index} of ${id} failed: ${errorMessage(err)}`);
                }
            }
            return out;
        }));
        this.sendUiEvent({
            uri: id,
            method: 'return_column_profiles',
            params: { callback_id: params.callback_id, profiles },
        });
    }

    private async exportDataSelection(
        ds: Dataset,
        selection: TableSelection,
        format: ExportFormat
    ): Promise<{ data: string; format: ExportFormat }> {
        const all = span(0, ds.result.columns.length - 1);
        let rows: KernelRows | undefined;
        let columns: number[];
        switch (selection.kind) {
            case 'single_cell': {
                // A single cell is copied as its bare value.
                const { row_index, column_index } = selection.selection;
                const page = await this.call<{ columns: (string | null)[][] }>(ds.sessionId, 'result_page', {
                    view: ds.result.view,
                    rows: { indices: [row_index] },
                    columns: [column_index],
                });
                return { data: page.columns[0]?.[0] ?? '', format };
            }
            case 'cell_range': {
                const s = selection.selection;
                rows = { first: s.first_row_index, last: s.last_row_index };
                columns = span(s.first_column_index, s.last_column_index);
                break;
            }
            case 'cell_indices':
                rows = { indices: selection.selection.row_indices };
                columns = selection.selection.column_indices;
                break;
            case 'column_range':
                columns = span(selection.selection.first_index, selection.selection.last_index);
                break;
            case 'row_range':
                rows = { first: selection.selection.first_index, last: selection.selection.last_index };
                columns = all;
                break;
            case 'column_indices':
                columns = selection.selection.indices;
                break;
            case 'row_indices':
                rows = { indices: selection.selection.indices };
                columns = all;
                break;
            default:
                throw new Error(`Unknown selection kind ${JSON.stringify((selection as { kind: unknown }).kind)}`);
        }
        const { data } = await this.call<{ data: string }>(ds.sessionId, 'result_export', {
            view: ds.result.view,
            ...(rows ? { rows } : {}),
            columns,
            format,
        });
        return { data, format };
    }
}

/**
 * Register the backend with Positron. Returns undefined, registering nothing,
 * when this Positron has no extension-hosted Data Explorer backends.
 */
export function registerResultExplorer(
    api: positron.PositronApi,
    subscriptions: { dispose(): unknown }[],
    log: (message: string) => void
): ResultExplorer | undefined {
    const dataExplorer = (api as Partial<positron.PositronApi>).dataExplorer;
    if (!dataExplorer?.registerRpcHandler) {
        return undefined;
    }
    // The handler is built before the registration it sends events through.
    // eslint-disable-next-line prefer-const
    let registration: positron.DataExplorerRpcSession | undefined;
    const explorer = new ResultExplorer(event => registration?.sendUiEvent(event), log);
    registration = dataExplorer.registerRpcHandler(PROVIDER_ID, {
        handleRpc: request => explorer.handleRpc(request),
        closeDataset: id => explorer.closeDataset(id),
    });
    subscriptions.push(registration);
    return explorer;
}
