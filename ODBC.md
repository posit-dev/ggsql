# Positron Connections Pane Integration for ggsql

## Context

ggsql's Jupyter kernel (`ggsql-jupyter`) and VS Code extension (`ggsql-vscode`) currently have no integration with Positron's Connections pane. The kernel is hardcoded to `duckdb://memory` with no way to configure the database connection. This plan adds:

1. **Connection comm protocol** in the kernel — so database schemas appear in the Connections pane
2. **Connection drivers** in the extension — so users can create connections via the "New Connection" dialog
3. **Generic ODBC reader** in the library — supporting Snowflake (with Workbench credentials), PostgreSQL, SQL Server, etc.
4. **Dynamic connection switching** in the kernel — via meta-commands executed from the connection dialog

## Architecture Overview

```
New Connection Dialog                ggsql-jupyter Kernel
  (ggsql-vscode)                      (Rust)
       │                                  │
       │ generateCode() →                 │
       │ "-- @connect: odbc://snowflake?…" │
       │                                  │
       │ connect() →                      │
       │ positron.runtime.executeCode()───┤
       │                                  │ detect meta-command
       │                                  │ create OdbcReader
       │                                  │ open positron.connection comm
       │                                  │
       │              ◄── comm_open ──────┤ (kernel initiates)
       │                                  │
  Connections Pane                        │
       │── list_objects([]) ─────────────►│ SELECT … FROM information_schema
       │◄─ [{name:"public",kind:"schema"}]│
       │── list_fields([schema,table]) ──►│ SELECT … FROM information_schema.columns
       │◄─ [{name:"id",dtype:"integer"}]  │
```

Key insight: The **kernel** opens the `positron.connection` comm (backend-initiated), unlike variables/ui comms which are frontend-initiated.

---

## Part 1: ODBC Reader (`src/reader/odbc.rs`)

### New file: `src/reader/odbc.rs`

Generic ODBC reader using `odbc-api` crate. Implements `Reader` trait.

**Connection string format**: `odbc://` prefix + raw ODBC connection string (no URI parsing of the payload)
- `odbc://Driver=Snowflake;Server=myaccount.snowflakecomputing.com;Warehouse=WH` — Snowflake
- `odbc://Driver={PostgreSQL};Server=localhost;Database=mydb` — PostgreSQL
- The extension's driver dialogs build the ODBC string in `generateCode()` and prefix it with `odbc://`
- Parsing: strip `odbc://` prefix, pass remainder directly to `SQLDriverConnect`

**Core implementation**:
- `OdbcReader::from_connection_string(uri)` — parse URI, detect credentials, connect
- `execute_sql(&self, sql)` — execute via `connection.execute()`, convert cursor → DataFrame
- Cursor → DataFrame conversion: iterate ODBC columnar buffers, map ODBC types to Polars types
- `register()` returns error (ODBC doesn't support temp table registration easily)
- `dialect()` returns dialect variant detected from DBMS info

**Snowflake Workbench credential detection** (per `~/work/positron/CONNECTIONS.md`):
When `OdbcReader` sees `Driver=Snowflake` in the connection string and no `Token=` is present:
1. Read `SNOWFLAKE_HOME` env var
2. If path contains `"posit-workbench"`, parse `$SNOWFLAKE_HOME/connections.toml`
3. Extract `account` + `token` from `[workbench]` section
4. Inject `Authenticator=oauth;Token=<token>` into the connection string before connecting
5. If no Workbench credentials found, connect as-is (user may have specified auth in the string)

**Credential storage**: Trust Positron's secret storage — the full `-- @connect:` meta-command (including any credentials in the ODBC string) is stored in the `code` field of the connection comm metadata. Positron persists this in encrypted workspace secret storage for reconnection.

**OdbcDialect**: Implements `SqlDialect` with a variant enum (Generic, Snowflake, PostgreSQL) detected from DBMS metadata at connection time.

### Modify: `src/reader/mod.rs`
- Add `#[cfg(feature = "odbc")] pub mod odbc;` and re-export `OdbcReader`
- Remove `where Self: Sized` bound from `fn execute()`

### Modify: `src/reader/connection.rs`
- Add `ODBC(String)` variant to `ConnectionInfo` enum
- Parse `odbc://` prefix in `parse_connection_string()`

### Modify: `src/execute/mod.rs`
- Change `prepare_data_with_reader<R: Reader>(query: &str, reader: &R)` → `prepare_data_with_reader(query: &str, reader: &dyn Reader)`
- This is safe: all methods called on reader (`execute_sql`, `dialect`, `register`, `unregister`) are object-safe. `materialize_ctes` already takes `&dyn Reader`.

### Modify: `src/Cargo.toml`
- Add feature: `odbc = ["dep:odbc-api", "dep:toml"]`
- Add dependencies: `odbc-api = { version = "21", optional = true }`, `toml = { version = "0.8", optional = true }`
- Add `"odbc"` to `all-readers` feature list

---

## Part 2: Kernel Connection Comm Protocol (`ggsql-jupyter/`)

### New file: `ggsql-jupyter/src/connection.rs`

Module for database schema introspection via the reader. All methods query `information_schema` using `reader.execute_sql()`.

**Methods**:
- `list_objects(reader, path) -> Vec<ObjectSchema>`:
  - Depth depends on dialect — `SqlDialect::has_catalogs()` (true for Snowflake, false for DuckDB/Postgres)
  - **Without catalogs** (DuckDB, PostgreSQL):
    - `[]` → query `information_schema.schemata` → return schemas
    - `[schema]` → query `information_schema.tables WHERE table_schema = '<escaped>'` → return tables/views
  - **With catalogs** (Snowflake):
    - `[]` → query `SHOW DATABASES` or `information_schema.schemata` grouped by catalog → return catalogs with `kind = "catalog"`
    - `[catalog]` → query `information_schema.schemata WHERE catalog_name = '<escaped>'` → return schemas
    - `[catalog, schema]` → query `information_schema.tables WHERE table_catalog = '<escaped>' AND table_schema = '<escaped>'` → return tables/views
- `list_fields(reader, path) -> Vec<FieldSchema>`:
  - **Without catalogs**: `[schema, table]` → query `information_schema.columns`
  - **With catalogs**: `[catalog, schema, table]` → query `information_schema.columns` with catalog filter
- `contains_data(path) -> bool`: true when last element has `kind` == "table" or "view"
- **SQL safety**: All interpolated identifiers use standard quote-escaping (`'` → `''`) via a shared `escape_sql_string()` helper
- `get_icon(path) -> String`: return empty string (let Positron use defaults)
- `preview_object(path)`: stub — return null (Data Explorer comm is a separate future feature)
- `get_metadata(reader_uri, name) -> MetadataSchema`: return connection metadata

**Dialect differences**: DuckDB's default schema is `main` (not `public`). Snowflake has a catalog→schema→table hierarchy. The `SqlDialect` trait gets new optional methods:
- `has_catalogs() -> bool` — false by default, true for Snowflake
- `schema_list_query() -> &str` — override for backends that don't support `information_schema.schemata`
- `default_schema() -> &str` — `"main"` for DuckDB, `"public"` for PostgreSQL, etc.

### Modify: `ggsql-jupyter/src/kernel.rs`

**Add connection comm tracking**:
```rust
connection_comm_id: Option<String>,
```

**Opening the comm** (kernel-initiated, sent on iopub after a successful `-- @connect:`):
```rust
// Send comm_open on iopub with target_name = "positron.connection"
self.send_iopub("comm_open", json!({
    "comm_id": new_uuid,
    "target_name": "positron.connection",
    "data": {
        "name": display_name,     // e.g. "DuckDB (memory)" or "Snowflake (myaccount)"
        "language_id": "ggsql",
        "host": host,             // e.g. "memory" or "myaccount.snowflakecomputing.com"
        "type": type_name,        // e.g. "DuckDB" or "Snowflake"
        "code": meta_command      // e.g. "-- @connect: duckdb://memory"
    }
}), parent).await?;
```

**Handle incoming comm_msg** for connection comm (JSON-RPC methods from Positron):
- Route `list_objects`, `list_fields`, `contains_data`, `get_icon`, `get_metadata` to `connection.rs` functions
- `preview_object`: stub — return null (full Data Explorer comm is a separate future feature)
- Send JSON-RPC responses back on shell via `send_shell_reply("comm_msg", ...)`

**Handle comm_close**: clear `connection_comm_id`

**Update comm_info_request**: include connection comm in response

**Open connection comm on startup**: After kernel initializes with default DuckDB reader, automatically open a `positron.connection` comm so the Connections pane shows the default database immediately.
- Use `create_message(..., None)` for the no-parent startup comm_open (same pattern as `send_status_initial` at kernel.rs:749)
- Add `send_iopub_no_parent()` helper (or generalize `send_iopub` to accept `Option<&JupyterMessage>`)

**Replacing connection comms on reader switch**: When `-- @connect:` switches readers:
1. If `connection_comm_id` is `Some`, send `comm_close` on iopub for the old comm ID first
2. Clear `connection_comm_id`
3. Open a new comm with a fresh UUID
4. This ensures no stale comm IDs linger and the Connections pane sees the old connection as disconnected + the new one as active

### Modify: `ggsql-jupyter/src/executor.rs`

**Make reader swappable**:
- Change `reader: DuckDBReader` → `reader: Box<dyn Reader + Send + Sync>`
- Add `pub fn swap_reader(&mut self, new_reader: Box<dyn Reader + Send + Sync>)`
- Add `pub fn reader(&self) -> &dyn Reader` accessor for connection.rs queries
- For visualization execution: call `self.reader.execute(query)` directly on the `Box<dyn Reader>`

**Add meta-command handling**:
- In `execute()`, check if code starts with `-- @connect: `
- Parse the connection URI from the meta-command
- Call `create_reader(uri)` (shared function) to build the new reader
- Swap the reader via `swap_reader()`
- Return a new `ExecutionResult::ConnectionChanged { uri, display_name }` variant

**New `create_reader(uri)` function** (in executor.rs or a new `reader_factory.rs`):
- Parse connection string using `ggsql::reader::connection::parse_connection_string()`
- Match on `ConnectionInfo` variant to construct appropriate reader
- Feature-gated: DuckDB (default), SQLite (optional), ODBC (optional)

### Modify: `ggsql-jupyter/src/main.rs`

- Add `--reader` CLI arg (default: `"duckdb://memory"`)
- Pass the reader URI to `KernelServer::new(connection, reader_uri)`
- Kernel creates initial reader from this URI

### Modify: `ggsql-jupyter/src/lib.rs`

- Add `mod connection;`

### Modify: `ggsql-jupyter/Cargo.toml`

Current: `ggsql = { workspace = true, features = ["duckdb", "vegalite"] }` — only DuckDB.

Add feature flags:
```toml
[features]
default = []
sqlite = ["ggsql/sqlite"]
odbc = ["ggsql/odbc"]
all-readers = ["sqlite", "odbc"]
```

Update ggsql dep: `ggsql = { workspace = true, features = ["duckdb", "vegalite"] }` stays as default (DuckDB always available).

**`create_reader()` runtime error handling**: When `-- @connect:` requests a reader that isn't compiled in, return a clear error message to the user via execute_reply:
```
Error: SQLite support is not compiled into this ggsql-jupyter binary.
Rebuild with: cargo build --features sqlite
```
This uses `#[cfg(feature = "...")]` branches with a fallback error arm per reader type.

---

## Part 3: VS Code Extension Connection Drivers (`ggsql-vscode/`)

### New file: `ggsql-vscode/src/connections.ts`

**`createConnectionDrivers(): positron.ConnectionsDriver[]`**

Returns array of drivers to register. Each driver:
- `generateCode(inputs)` → returns `-- @connect: <uri>` meta-command string
- `connect(code)` → calls `positron.runtime.executeCode('ggsql', code, false)` to send the meta-command to the running kernel

**DuckDB driver** (`driverId: 'ggsql-duckdb'`):
- Inputs: `database` (string, optional — empty = in-memory)
- generateCode: `-- @connect: duckdb://memory` or `-- @connect: duckdb://<path>`

**Snowflake driver** (`driverId: 'ggsql-snowflake'`):
- Inputs: `account` (string, required), `warehouse` (string, required), `database` (string, optional), `schema` (string, optional)
- generateCode: builds full ODBC string e.g. `-- @connect: odbc://Driver=Snowflake;Server=<account>.snowflakecomputing.com;Warehouse=<warehouse>`

**Generic ODBC driver** (`driverId: 'ggsql-odbc'`):
- Inputs: `connection_string` (string, required — raw ODBC connection string)
- generateCode: `-- @connect: odbc://<connection_string>`

### Modify: `ggsql-vscode/src/extension.ts`

In `activate()`, after registering the runtime manager:
```typescript
import { createConnectionDrivers } from './connections';
// ...
const drivers = createConnectionDrivers();
for (const driver of drivers) {
    context.subscriptions.push(positronApi.connections.registerConnectionDriver(driver));
}
```

### Modify: `ggsql-vscode/src/manager.ts`

- Update `createKernelSpec()` to accept optional `readerUri` parameter
- Pass `--reader <uri>` in spawn args when `readerUri` is provided
- Add `getActiveSession()` method so `connect()` can check if a kernel is running

---

## Implementation Order

### Phase 1: Kernel meta-commands and dynamic reader switching
1. Modify `executor.rs` — make reader swappable, add meta-command detection, add `create_reader()`
2. Modify `main.rs` — add `--reader` CLI arg, pass to executor
3. Modify `kernel.rs` — handle `ConnectionChanged` result from executor
4. Test: start kernel with `--reader duckdb://memory`, verify meta-command works

### Phase 2: Connection comm protocol
5. Create `connection.rs` — schema introspection via information_schema
6. Modify `kernel.rs` — open `positron.connection` comm on startup and after `-- @connect:`, handle incoming JSON-RPC methods
7. Test: start kernel in Positron, verify Connections pane shows DuckDB schema

### Phase 3: ODBC reader
8. Create `src/reader/odbc.rs` — generic ODBC reader with cursor→DataFrame conversion
9. Add Workbench Snowflake credential detection
10. Modify `connection.rs`, `mod.rs`, `Cargo.toml` for ODBC feature
11. Test: connect to local ODBC data source, verify queries work

### Phase 4: Extension connection drivers
12. Create `ggsql-vscode/src/connections.ts` — DuckDB, Snowflake, generic ODBC drivers
13. Modify `extension.ts` — register drivers on activation
14. Test: open New Connection dialog, create DuckDB connection, verify Connections pane updates

### Phase 5: Integration & polish
15. End-to-end test: New Connection dialog → kernel connection → Connections pane browsing
16. Handle edge cases: connection failures, reader not compiled in, comm lifecycle

---

## Verification

1. **Unit tests**: Meta-command parsing, ODBC URI parsing, Workbench credential detection, schema introspection queries
2. **Integration test**: Start kernel with `--reader duckdb://memory`, execute `-- @connect: duckdb://memory`, verify comm_open message on iopub
3. **Manual Positron test**: Open ggsql session → Connections pane shows DuckDB → expand to see schemas/tables/columns → New Connection dialog → create Snowflake connection → Connections pane updates
4. **Existing tests**: Run `cargo test` to ensure no regressions in parser/reader/writer

## Key files to modify

| File | Change |
|------|--------|
| `src/execute/mod.rs` | Change `prepare_data_with_reader` to `&dyn Reader` |
| `src/reader/mod.rs` | Remove `Self: Sized` from `execute()`, add odbc module |
| `src/reader/odbc.rs` | **NEW** — Generic ODBC reader |
| `src/reader/connection.rs` | Add ODBC variant |
| `src/Cargo.toml` | Add odbc feature + deps |
| `ggsql-jupyter/src/connection.rs` | **NEW** — Schema introspection |
| `ggsql-jupyter/src/kernel.rs` | Connection comm protocol |
| `ggsql-jupyter/src/executor.rs` | Dynamic reader switching, meta-commands |
| `ggsql-jupyter/src/main.rs` | `--reader` CLI arg |
| `ggsql-jupyter/src/lib.rs` | Add connection module |
| `ggsql-jupyter/Cargo.toml` | Add odbc feature |
| `ggsql-vscode/src/connections.ts` | **NEW** — Connection drivers |
| `ggsql-vscode/src/extension.ts` | Register connection drivers |
| `ggsql-vscode/src/manager.ts` | Pass reader URI to kernel |
