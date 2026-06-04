// Converters
export { convert_csv } from "./csv";
export { convert_parquet } from "./parquet";

// Extension loading
export { initExtensionLoader, installExtension } from "./extensions";

// Types
export interface ColumnDescriptor {
  name: string;
  type: ColumnType;
  // "binary" columns carry one Uint8Array per row; all others use the typed
  // forms below.
  values: Float64Array | Uint8Array | string[] | Uint8Array[];
  nulls: Uint8Array;
}

export type ColumnType =
  | "f64"
  | "i64"
  | "bool"
  | "date"
  | "datetime"
  | "string"
  | "binary";

export const EPOCH = Date.UTC(1970, 0, 1);
export const MS_PER_DAY = 86400000;
