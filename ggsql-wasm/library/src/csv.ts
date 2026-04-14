import type { ColumnDescriptor, ColumnType } from "./index";

/**
 * Convert CSV bytes to column descriptors.
 * Synchronous — no external dependencies.
 */
export function convert_csv(bytes: Uint8Array): ColumnDescriptor[] {
  const text = new TextDecoder().decode(bytes);
  const lines = parseCSVLines(text);

  if (lines.length < 2) return [];

  const headers = lines[0];
  const nCols = headers.length;
  const nRows = lines.length - 1;

  // Collect raw string values per column
  const rawCols: (string | null)[][] = [];
  for (let c = 0; c < nCols; c++) {
    rawCols.push(new Array(nRows));
  }

  for (let r = 0; r < nRows; r++) {
    const row = lines[r + 1];
    for (let c = 0; c < nCols; c++) {
      const val = c < row.length ? row[c] : "";
      rawCols[c][r] = val === "" ? null : val;
    }
  }

  // Per-column type inference and conversion
  const columns: ColumnDescriptor[] = [];
  for (let c = 0; c < nCols; c++) {
    const raw = rawCols[c];
    const colType = inferCSVColumnType(raw);
    columns.push(buildCSVColumn(headers[c], raw, colType));
  }

  return columns;
}

function inferCSVColumnType(values: (string | null)[]): ColumnType {
  let allInt = true;
  let allNum = true;
  let allBool = true;

  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v === null) continue;

    const lower = v.toLowerCase();
    if (lower !== "true" && lower !== "false") allBool = false;

    const num = Number(v);
    if (v === "" || isNaN(num)) {
      allNum = false;
      allInt = false;
    } else {
      if (!Number.isSafeInteger(num)) allInt = false;
    }
  }

  if (allBool) return "bool";
  if (allInt) return "i64";
  if (allNum) return "f64";
  return "string";
}

function buildCSVColumn(
  name: string,
  rawStrings: (string | null)[],
  colType: ColumnType,
): ColumnDescriptor {
  const len = rawStrings.length;
  const nulls = new Uint8Array(len);

  if (colType === "f64" || colType === "i64") {
    const values = new Float64Array(len);
    for (let i = 0; i < len; i++) {
      if (rawStrings[i] === null) {
        values[i] = 0;
        nulls[i] = 0;
      } else {
        values[i] = Number(rawStrings[i]);
        nulls[i] = 1;
      }
    }
    return { name, type: colType, values, nulls };
  }

  if (colType === "bool") {
    const values = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      if (rawStrings[i] === null) {
        values[i] = 0;
        nulls[i] = 0;
      } else {
        values[i] = rawStrings[i]!.toLowerCase() === "true" ? 1 : 0;
        nulls[i] = 1;
      }
    }
    return { name, type: colType, values, nulls };
  }

  // string
  const values: string[] = [];
  for (let i = 0; i < len; i++) {
    if (rawStrings[i] === null) {
      values.push("");
      nulls[i] = 0;
    } else {
      values.push(rawStrings[i]!);
      nulls[i] = 1;
    }
  }
  return { name, type: "string", values, nulls };
}

/**
 * Parse CSV text into an array of rows (each row is an array of fields).
 * Handles quoted fields, embedded commas, and embedded newlines.
 */
function parseCSVLines(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;
  let i = 0;

  while (i < text.length) {
    const ch = text[i];

    if (inQuotes) {
      if (ch === '"') {
        if (i + 1 < text.length && text[i + 1] === '"') {
          field += '"';
          i += 2;
        } else {
          inQuotes = false;
          i++;
        }
      } else {
        field += ch;
        i++;
      }
    } else {
      if (ch === '"') {
        inQuotes = true;
        i++;
      } else if (ch === ",") {
        row.push(field);
        field = "";
        i++;
      } else if (ch === "\r") {
        row.push(field);
        field = "";
        rows.push(row);
        row = [];
        i++;
        if (i < text.length && text[i] === "\n") i++;
      } else if (ch === "\n") {
        row.push(field);
        field = "";
        rows.push(row);
        row = [];
        i++;
      } else {
        field += ch;
        i++;
      }
    }
  }

  // Handle last field/row
  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }

  return rows;
}
