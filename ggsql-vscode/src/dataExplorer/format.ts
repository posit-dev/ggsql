/*
 * Cells from the kernel as Positron's `ColumnValue`s.
 */

import type { ColumnKind, ColumnValue } from './protocol';

/** Positron's codes for values that are not shown as text. */
export const SpecialValue = {
    Null: 0,
    NaN: 2,
    Inf: 10,
    NegInf: 11,
} as const;

/**
 * A kernel cell as Positron shows it: the text, or a special value's code.
 * Only a floating column's text can spell a non-finite number; in any other
 * column "NaN" is just text.
 */
export function cellValue(cell: string | null, kind: ColumnKind): ColumnValue {
    if (cell === null) {
        return SpecialValue.Null;
    }
    if (kind === 'floating') {
        switch (cell.toLowerCase()) {
            case 'nan':
                return SpecialValue.NaN;
            case 'inf':
            case 'infinity':
                return SpecialValue.Inf;
            case '-inf':
            case '-infinity':
                return SpecialValue.NegInf;
        }
    }
    return cell;
}

/** `first..=last`, empty when `last < first`. */
export function span(first: number, last: number): number[] {
    const out: number[] = [];
    for (let i = first; i <= last; i++) {
        out.push(i);
    }
    return out;
}

/** A thrown value's message: kernel errors arrive as plain objects. */
export function errorMessage(err: unknown): string {
    if (err instanceof Error) {
        return err.message;
    }
    if (typeof err === 'object' && err !== null && typeof (err as { message?: unknown }).message === 'string') {
        return (err as { message: string }).message;
    }
    return String(err);
}
