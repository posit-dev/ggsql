import type { ColumnDescriptor, ColumnType } from "./index";
import { EPOCH, MS_PER_DAY } from "./index";
import { parquetReadObjects } from "hyparquet";

/**
 * Convert Parquet bytes to column descriptors.
 * Dynamically imports hyparquet.
 */
export async function convert_parquet(
  bytes: Uint8Array,
): Promise<ColumnDescriptor[]> {
  const buffer = bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  );

  const asyncBuffer = {
    byteLength: buffer.byteLength,
    slice: (start: number, end: number) =>
      Promise.resolve(buffer.slice(start, end) as ArrayBuffer),
  };

  const rows: Record<string, unknown>[] = await parquetReadObjects({
    file: asyncBuffer,
  });

  if (rows.length === 0) return [];

  const colNames = Object.keys(rows[0]);
  const columns: ColumnDescriptor[] = [];

  for (const colName of colNames) {
    const rawValues = rows.map((row) => row[colName]);
    columns.push(buildColumn(colName, rawValues));
  }

  return columns;
}

function inferColumnType(values: unknown[]): ColumnType {
  let hasNumber = false;
  let hasBool = false;
  let hasDate = false;
  let allSafeInt = true;
  let allMidnight = true;

  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v === null || v === undefined) continue;

    if (v instanceof Date) {
      hasDate = true;
      if (
        v.getUTCHours() !== 0 ||
        v.getUTCMinutes() !== 0 ||
        v.getUTCSeconds() !== 0 ||
        v.getUTCMilliseconds() !== 0
      ) {
        allMidnight = false;
      }
    } else if (typeof v === "boolean") {
      hasBool = true;
    } else if (typeof v === "number") {
      hasNumber = true;
      if (!Number.isSafeInteger(v)) allSafeInt = false;
    } else if (typeof v === "bigint") {
      hasNumber = true;
      // bigint values will be converted to Number, keep as i64
    } else {
      return "string";
    }
  }

  if (hasDate) return allMidnight ? "date" : "datetime";
  if (hasBool && !hasNumber) return "bool";
  if (hasNumber) return allSafeInt ? "i64" : "f64";
  return "string";
}

function buildColumn(name: string, rawValues: unknown[]): ColumnDescriptor {
  const len = rawValues.length;
  const nulls = new Uint8Array(len);
  const type = inferColumnType(rawValues);

  if (type === "f64" || type === "i64") {
    const values = new Float64Array(len);
    for (let i = 0; i < len; i++) {
      const v = rawValues[i];
      if (v === null || v === undefined) {
        values[i] = 0;
        nulls[i] = 0;
      } else {
        values[i] = Number(v);
        nulls[i] = 1;
      }
    }
    return { name, type, values, nulls };
  }

  if (type === "bool") {
    const values = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      const v = rawValues[i];
      if (v === null || v === undefined) {
        values[i] = 0;
        nulls[i] = 0;
      } else {
        values[i] = v ? 1 : 0;
        nulls[i] = 1;
      }
    }
    return { name, type, values, nulls };
  }

  if (type === "date") {
    const values = new Float64Array(len);
    for (let i = 0; i < len; i++) {
      const v = rawValues[i] as Date | null | undefined;
      if (v === null || v === undefined) {
        values[i] = 0;
        nulls[i] = 0;
      } else {
        values[i] = Math.floor((v.getTime() - EPOCH) / MS_PER_DAY);
        nulls[i] = 1;
      }
    }
    return { name, type, values, nulls };
  }

  if (type === "datetime") {
    const values = new Float64Array(len);
    for (let i = 0; i < len; i++) {
      const v = rawValues[i] as Date | null | undefined;
      if (v === null || v === undefined) {
        values[i] = 0;
        nulls[i] = 0;
      } else {
        values[i] = v.getTime();
        nulls[i] = 1;
      }
    }
    return { name, type, values, nulls };
  }

  // string
  const values: string[] = [];
  for (let i = 0; i < len; i++) {
    const v = rawValues[i];
    if (v === null || v === undefined) {
      values.push("");
      nulls[i] = 0;
    } else {
      values.push(String(v));
      nulls[i] = 1;
    }
  }
  return { name, type, values, nulls };
}
