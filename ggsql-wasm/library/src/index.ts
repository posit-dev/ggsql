// Converters
export { convert_csv } from "./csv";
export { convert_parquet } from "./parquet";

// Types
export interface ColumnDescriptor {
  name: string;
  type: ColumnType;
  values: Float64Array | Uint8Array | string[];
  nulls: Uint8Array;
}

export type ColumnType =
  | "f64"
  | "i64"
  | "bool"
  | "date"
  | "datetime"
  | "string";

export const EPOCH = Date.UTC(1970, 0, 1);
export const MS_PER_DAY = 86400000;
