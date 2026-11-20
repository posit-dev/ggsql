/*
 * Positron's column profile requests in the kernel's terms, and the kernel's
 * profiles in Positron's `ColumnProfileResult` shape.
 */

import type {
    ColumnFrequencyTableParams,
    ColumnHistogram,
    ColumnHistogramParams,
    ColumnKind,
    ColumnProfileResult,
    ColumnProfileSpec,
    ColumnSummaryStats,
    KernelProfile,
    KernelProfileRequest,
} from './protocol';

/** Which Positron key each part of one kernel request answers. */
export interface ProfileCall {
    request: KernelProfileRequest;
    histogramKey?: 'small_histogram' | 'large_histogram';
    frequencyKey?: 'small_frequency_table' | 'large_frequency_table';
}

/**
 * The kernel requests for one column's specs. The kernel computes one
 * histogram and one frequency table per call, so the large variants, if
 * asked for, get a call of their own.
 */
export function toProfileCalls(specs: ColumnProfileSpec[]): ProfileCall[] {
    const small: ProfileCall = { request: {} };
    const large: ProfileCall = { request: {} };
    for (const spec of specs) {
        switch (spec.profile_type) {
            case 'null_count':
                small.request.null_count = true;
                break;
            case 'summary_stats':
                small.request.summary = true;
                break;
            case 'small_histogram':
            case 'large_histogram': {
                const call = spec.profile_type === 'small_histogram' ? small : large;
                const params = spec.params as ColumnHistogramParams | undefined;
                call.request.histogram = {
                    bins: params?.num_bins ?? 20,
                    quantiles: params?.quantiles ?? [],
                };
                call.histogramKey = spec.profile_type;
                break;
            }
            case 'small_frequency_table':
            case 'large_frequency_table': {
                const call = spec.profile_type === 'small_frequency_table' ? small : large;
                const params = spec.params as ColumnFrequencyTableParams | undefined;
                call.request.frequency = { limit: params?.limit ?? 8 };
                call.frequencyKey = spec.profile_type;
                break;
            }
        }
    }
    return [small, large].filter(c => Object.keys(c.request).length > 0);
}

function str(v: unknown): string | undefined {
    return typeof v === 'string' ? v : undefined;
}

function num(v: unknown): number | undefined {
    return typeof v === 'number' ? v : undefined;
}

/** The kernel's summary of a column of `kind` as Positron's summary stats. */
export function toSummaryStats(kind: ColumnKind, summary: Record<string, unknown>): ColumnSummaryStats {
    switch (kind) {
        case 'integer':
        case 'floating':
            return {
                type_display: kind,
                number_stats: {
                    min_value: str(summary.min),
                    max_value: str(summary.max),
                    mean: str(summary.mean),
                    median: str(summary.median),
                    stdev: str(summary.stdev),
                },
            };
        case 'boolean':
            return {
                type_display: kind,
                boolean_stats: {
                    true_count: num(summary.true_count) ?? 0,
                    false_count: num(summary.false_count) ?? 0,
                },
            };
        case 'string':
            return {
                type_display: kind,
                string_stats: {
                    num_unique: num(summary.num_unique) ?? 0,
                    num_empty: num(summary.num_empty) ?? 0,
                },
            };
        case 'date':
        case 'datetime': {
            const stats = {
                min_date: str(summary.min),
                max_date: str(summary.max),
                num_unique: num(summary.num_unique),
            };
            return kind === 'date'
                ? { type_display: kind, date_stats: stats }
                : { type_display: kind, datetime_stats: stats };
        }
        default:
            return { type_display: kind };
    }
}

function toHistogram(h: NonNullable<KernelProfile['histogram']>): ColumnHistogram {
    return {
        bin_edges: h.bin_edges,
        bin_counts: h.bin_counts,
        // The kernel's quantiles are computed in SQL over every row, so exact.
        quantiles: h.quantiles.map(({ q, value }) => ({ q, value, exact: true })),
    };
}

/** Merge one kernel profile into the Positron result, under the keys `call` asked for. */
export function addProfile(
    out: ColumnProfileResult,
    call: ProfileCall,
    profile: KernelProfile,
    kind: ColumnKind
): void {
    if (profile.null_count !== undefined) {
        out.null_count = profile.null_count;
    }
    if (profile.summary !== undefined) {
        out.summary_stats = toSummaryStats(kind, profile.summary);
    }
    if (profile.histogram && call.histogramKey) {
        out[call.histogramKey] = toHistogram(profile.histogram);
    }
    if (profile.frequency && call.frequencyKey) {
        out[call.frequencyKey] = {
            values: profile.frequency.values,
            counts: profile.frequency.counts,
            other_count: profile.frequency.other_count,
        };
    }
}
