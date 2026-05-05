//! Safe Rust wrappers around ODBC handles.
//!
//! Provides `Environment`, `Connection`, `Statement`, and `PreparedStatement`
//! types that own ODBC handles and call through the runtime-loaded FFI.

use super::ffi::*;
use crate::{GgsqlError, Result};
use std::sync::OnceLock;

// ============================================================================
// Diagnostic helpers
// ============================================================================

fn extract_diagnostic(handle_type: SqlSmallInt, handle: SqlHandle) -> String {
    let f = fns();
    let mut state = [0u8; 6];
    let mut native_error: SqlInteger = 0;
    let mut buf = vec![0u8; 512];
    let mut text_len: SqlSmallInt = 0;

    let rc = unsafe {
        (f.SQLGetDiagRec)(
            handle_type,
            handle,
            1,
            state.as_mut_ptr(),
            &mut native_error,
            buf.as_mut_ptr(),
            buf.len() as SqlSmallInt,
            &mut text_len,
        )
    };

    if !succeeded(rc) {
        return "Unknown ODBC error (no diagnostic record)".to_string();
    }

    // Retry with larger buffer if truncated
    if text_len as usize >= buf.len() {
        buf.resize(text_len as usize + 1, 0);
        unsafe {
            (f.SQLGetDiagRec)(
                handle_type,
                handle,
                1,
                state.as_mut_ptr(),
                &mut native_error,
                buf.as_mut_ptr(),
                buf.len() as SqlSmallInt,
                &mut text_len,
            );
        }
    }

    let state_str = std::str::from_utf8(&state[..5]).unwrap_or("?????");
    let msg = std::str::from_utf8(&buf[..text_len as usize]).unwrap_or("(invalid UTF-8)");
    format!("[{}] {}", state_str, msg)
}

fn check(rc: SqlReturn, handle_type: SqlSmallInt, handle: SqlHandle, context: &str) -> Result<()> {
    match rc {
        SQL_SUCCESS | SQL_SUCCESS_WITH_INFO => Ok(()),
        SQL_NO_DATA => Ok(()),
        _ => {
            let diag = extract_diagnostic(handle_type, handle);
            Err(GgsqlError::ReaderError(format!("{}: {}", context, diag)))
        }
    }
}

/// Check if the SQLSTATE from the last operation matches a specific code.
fn sqlstate_is(handle_type: SqlSmallInt, handle: SqlHandle, expected: &[u8; 5]) -> bool {
    let f = fns();
    let mut state = [0u8; 6];
    let mut native_error: SqlInteger = 0;
    let mut text_len: SqlSmallInt = 0;
    let rc = unsafe {
        (f.SQLGetDiagRec)(
            handle_type,
            handle,
            1,
            state.as_mut_ptr(),
            &mut native_error,
            std::ptr::null_mut(),
            0,
            &mut text_len,
        )
    };
    succeeded(rc) && state[..5] == expected[..]
}

// ============================================================================
// Environment
// ============================================================================

pub struct Environment {
    handle: SqlHEnv,
}

unsafe impl Send for Environment {}
unsafe impl Sync for Environment {}

impl Environment {
    fn new() -> Result<Self> {
        let f = fns();
        let mut handle = SQL_NULL_HANDLE;
        let rc = unsafe { (f.SQLAllocHandle)(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &mut handle) };
        if !succeeded(rc) {
            return Err(GgsqlError::ReaderError(
                "Failed to allocate ODBC environment handle".into(),
            ));
        }

        let rc = unsafe {
            (f.SQLSetEnvAttr)(
                handle,
                SQL_ATTR_ODBC_VERSION,
                SQL_OV_ODBC3_80 as SqlPointer,
                0,
            )
        };
        check(rc, SQL_HANDLE_ENV, handle, "Failed to set ODBC version")?;

        Ok(Environment { handle })
    }

    pub fn handle(&self) -> SqlHEnv {
        self.handle
    }
}

impl Drop for Environment {
    fn drop(&mut self) {
        let f = fns();
        unsafe { (f.SQLFreeHandle)(SQL_HANDLE_ENV, self.handle) };
    }
}

/// Global ODBC environment (singleton per process).
pub fn odbc_env() -> Result<&'static Environment> {
    static ENV: OnceLock<std::result::Result<Environment, String>> = OnceLock::new();
    let result = ENV.get_or_init(|| Environment::new().map_err(|e| e.to_string()));
    match result {
        Ok(env) => Ok(env),
        Err(e) => Err(GgsqlError::ReaderError(e.clone())),
    }
}

// ============================================================================
// Connection
// ============================================================================

pub struct Connection {
    handle: SqlHDbc,
}

unsafe impl Send for Connection {}

impl Connection {
    pub fn connect(env: &Environment, conn_str: &str) -> Result<Self> {
        let f = fns();
        let mut handle = SQL_NULL_HANDLE;
        let rc = unsafe { (f.SQLAllocHandle)(SQL_HANDLE_DBC, env.handle(), &mut handle) };
        if !succeeded(rc) {
            return Err(GgsqlError::ReaderError(
                "Failed to allocate ODBC connection handle".into(),
            ));
        }

        let conn_cstr = std::ffi::CString::new(conn_str)
            .map_err(|_| GgsqlError::ReaderError("Connection string contains null byte".into()))?;
        let rc = unsafe {
            (f.SQLDriverConnect)(
                handle,
                std::ptr::null_mut(), // no window handle
                conn_cstr.as_ptr() as *const SqlChar,
                conn_str.len() as SqlSmallInt,
                std::ptr::null_mut(), // no output buffer
                0,
                std::ptr::null_mut(),
                SQL_DRIVER_NOPROMPT,
            )
        };
        if !succeeded(rc) {
            let diag = extract_diagnostic(SQL_HANDLE_DBC, handle);
            unsafe { (f.SQLFreeHandle)(SQL_HANDLE_DBC, handle) };
            return Err(GgsqlError::ReaderError(format!(
                "ODBC connection failed: {}",
                diag
            )));
        }

        Ok(Connection { handle })
    }

    pub fn handle(&self) -> SqlHDbc {
        self.handle
    }

    /// Execute a SQL statement, returning a Statement if it produces a result set.
    pub fn execute(&self, sql: &str) -> Result<Option<Statement>> {
        let f = fns();
        let mut stmt_handle = SQL_NULL_HANDLE;
        let rc = unsafe { (f.SQLAllocHandle)(SQL_HANDLE_STMT, self.handle, &mut stmt_handle) };
        check(
            rc,
            SQL_HANDLE_DBC,
            self.handle,
            "Failed to allocate statement",
        )?;

        let sql_cstr = std::ffi::CString::new(sql)
            .map_err(|_| GgsqlError::ReaderError("SQL string contains null byte".into()))?;
        let rc = unsafe {
            (f.SQLExecDirect)(
                stmt_handle,
                sql_cstr.as_ptr() as *const SqlChar,
                sql.len() as SqlInteger,
            )
        };

        match rc {
            SQL_SUCCESS | SQL_SUCCESS_WITH_INFO => {}
            SQL_NO_DATA => {
                // DDL or statement with no result set
                unsafe { (f.SQLFreeHandle)(SQL_HANDLE_STMT, stmt_handle) };
                return Ok(None);
            }
            _ => {
                let diag = extract_diagnostic(SQL_HANDLE_STMT, stmt_handle);
                unsafe { (f.SQLFreeHandle)(SQL_HANDLE_STMT, stmt_handle) };
                return Err(GgsqlError::ReaderError(format!(
                    "ODBC execute failed: {}",
                    diag
                )));
            }
        }

        // Check if there's a result set
        let mut col_count: SqlSmallInt = 0;
        let rc = unsafe { (f.SQLNumResultCols)(stmt_handle, &mut col_count) };
        check(
            rc,
            SQL_HANDLE_STMT,
            stmt_handle,
            "Failed to get column count",
        )?;

        if col_count == 0 {
            unsafe { (f.SQLFreeHandle)(SQL_HANDLE_STMT, stmt_handle) };
            return Ok(None);
        }

        Ok(Some(Statement {
            handle: stmt_handle,
        }))
    }

    /// Prepare a SQL statement for repeated execution with parameters.
    pub fn prepare(&self, sql: &str) -> Result<PreparedStatement> {
        let f = fns();
        let mut stmt_handle = SQL_NULL_HANDLE;
        let rc = unsafe { (f.SQLAllocHandle)(SQL_HANDLE_STMT, self.handle, &mut stmt_handle) };
        check(
            rc,
            SQL_HANDLE_DBC,
            self.handle,
            "Failed to allocate statement",
        )?;

        let sql_cstr = std::ffi::CString::new(sql)
            .map_err(|_| GgsqlError::ReaderError("SQL string contains null byte".into()))?;
        let rc = unsafe {
            (f.SQLPrepare)(
                stmt_handle,
                sql_cstr.as_ptr() as *const SqlChar,
                sql.len() as SqlInteger,
            )
        };
        if !succeeded(rc) {
            let diag = extract_diagnostic(SQL_HANDLE_STMT, stmt_handle);
            unsafe { (f.SQLFreeHandle)(SQL_HANDLE_STMT, stmt_handle) };
            return Err(GgsqlError::ReaderError(format!(
                "ODBC prepare failed: {}",
                diag
            )));
        }

        Ok(PreparedStatement {
            handle: stmt_handle,
        })
    }

    /// Get DBMS name via SQLGetInfo.
    pub fn dbms_name(&self) -> Option<String> {
        let f = fns();
        let mut buf = vec![0u8; 256];
        let mut len: SqlSmallInt = 0;
        let rc = unsafe {
            (f.SQLGetInfo)(
                self.handle,
                SQL_DBMS_NAME,
                buf.as_mut_ptr() as SqlPointer,
                buf.len() as SqlSmallInt,
                &mut len,
            )
        };
        if succeeded(rc) && len > 0 {
            let s = std::str::from_utf8(&buf[..len as usize]).ok()?.to_string();
            Some(s)
        } else {
            None
        }
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        let f = fns();
        let rc = unsafe { (f.SQLDisconnect)(self.handle) };
        // If there's an open transaction, attempt rollback then retry
        if !succeeded(rc) && sqlstate_is(SQL_HANDLE_DBC, self.handle, b"25000") {
            unsafe {
                (f.SQLEndTran)(SQL_HANDLE_DBC, self.handle, SQL_ROLLBACK);
                (f.SQLDisconnect)(self.handle);
            }
        }
        unsafe { (f.SQLFreeHandle)(SQL_HANDLE_DBC, self.handle) };
    }
}

// ============================================================================
// Statement (result cursor)
// ============================================================================

pub struct Statement {
    handle: SqlHStmt,
}

impl Statement {
    pub fn handle(&self) -> SqlHStmt {
        self.handle
    }

    pub fn num_result_cols(&self) -> Result<usize> {
        let f = fns();
        let mut count: SqlSmallInt = 0;
        let rc = unsafe { (f.SQLNumResultCols)(self.handle, &mut count) };
        check(
            rc,
            SQL_HANDLE_STMT,
            self.handle,
            "Failed to get column count",
        )?;
        Ok(count as usize)
    }

    /// Describe column `col` (1-based).
    /// Returns (name, sql_data_type, column_size, decimal_digits, nullable).
    pub fn describe_col(
        &self,
        col: u16,
    ) -> Result<(String, SqlSmallInt, SqlULen, SqlSmallInt, bool)> {
        let f = fns();
        let mut name_buf = vec![0u8; 256];
        let mut name_len: SqlSmallInt = 0;
        let mut data_type: SqlSmallInt = 0;
        let mut col_size: SqlULen = 0;
        let mut decimal_digits: SqlSmallInt = 0;
        let mut nullable: SqlSmallInt = 0;

        let rc = unsafe {
            (f.SQLDescribeCol)(
                self.handle,
                col,
                name_buf.as_mut_ptr(),
                name_buf.len() as SqlSmallInt,
                &mut name_len,
                &mut data_type,
                &mut col_size,
                &mut decimal_digits,
                &mut nullable,
            )
        };
        check(
            rc,
            SQL_HANDLE_STMT,
            self.handle,
            "Failed to describe column",
        )?;

        let name = std::str::from_utf8(&name_buf[..name_len as usize])
            .unwrap_or("?")
            .to_string();

        Ok((
            name,
            data_type,
            col_size,
            decimal_digits,
            nullable != SQL_NO_NULLS,
        ))
    }

    /// Bind a column buffer for batch fetching (1-based column index).
    pub fn bind_col(
        &self,
        col: u16,
        c_type: SqlSmallInt,
        buffer: SqlPointer,
        buffer_len: SqlLen,
        indicator: *mut SqlLen,
    ) -> Result<()> {
        let f = fns();
        let rc = unsafe { (f.SQLBindCol)(self.handle, col, c_type, buffer, buffer_len, indicator) };
        check(rc, SQL_HANDLE_STMT, self.handle, "Failed to bind column")
    }

    /// Set a statement attribute.
    pub fn set_stmt_attr(
        &self,
        attribute: SqlInteger,
        value: SqlPointer,
        string_length: SqlInteger,
    ) -> Result<()> {
        let f = fns();
        let rc = unsafe { (f.SQLSetStmtAttr)(self.handle, attribute, value, string_length) };
        check(
            rc,
            SQL_HANDLE_STMT,
            self.handle,
            "Failed to set statement attribute",
        )
    }

    /// Set up batch fetching with the given batch size.
    /// Returns a mutable reference location for rows_fetched.
    pub fn setup_batch_fetch(&self, batch_size: usize) -> Result<()> {
        // Column-wise binding
        self.set_stmt_attr(SQL_ATTR_ROW_BIND_TYPE, SQL_BIND_BY_COLUMN as SqlPointer, 0)?;
        // Array size
        self.set_stmt_attr(SQL_ATTR_ROW_ARRAY_SIZE, batch_size as SqlPointer, 0)?;
        Ok(())
    }

    /// Set the rows-fetched pointer.
    ///
    /// # Safety
    /// `rows_fetched` must remain valid and pinned for the lifetime of the cursor.
    pub unsafe fn set_rows_fetched_ptr(&self, rows_fetched: *mut SqlULen) -> Result<()> {
        self.set_stmt_attr(SQL_ATTR_ROWS_FETCHED_PTR, rows_fetched as SqlPointer, 0)
    }

    /// Fetch the next batch of rows. Returns the ODBC return code.
    pub fn fetch_raw(&self) -> SqlReturn {
        let f = fns();
        unsafe { (f.SQLFetch)(self.handle) }
    }

    /// Unbind all columns.
    pub fn unbind_cols(&self) -> Result<()> {
        let f = fns();
        let rc = unsafe { (f.SQLFreeStmt)(self.handle, SQL_UNBIND) };
        check(rc, SQL_HANDLE_STMT, self.handle, "Failed to unbind columns")
    }
}

impl Drop for Statement {
    fn drop(&mut self) {
        let f = fns();
        let rc = unsafe { (f.SQLFreeHandle)(SQL_HANDLE_STMT, self.handle) };
        if !succeeded(rc) && !std::thread::panicking() {
            panic!(
                "SQLFreeHandle(STMT) failed: {}",
                extract_diagnostic(SQL_HANDLE_STMT, self.handle)
            );
        }
    }
}

// ============================================================================
// PreparedStatement (for bulk insert)
// ============================================================================

pub struct PreparedStatement {
    handle: SqlHStmt,
}

impl PreparedStatement {
    /// Bind a text parameter (1-based index).
    ///
    /// # Safety
    /// `value_ptr` and `indicator` must remain valid until execute() or reset_params().
    pub unsafe fn bind_text_parameter(
        &self,
        param_num: u16,
        value_ptr: *const u8,
        buffer_len: SqlLen,
        indicator: *mut SqlLen,
    ) -> Result<()> {
        let f = fns();
        let rc = unsafe {
            (f.SQLBindParameter)(
                self.handle,
                param_num,
                SQL_PARAM_INPUT,
                SQL_C_CHAR,
                SQL_VARCHAR,
                buffer_len as SqlULen,
                0,
                value_ptr as SqlPointer,
                buffer_len,
                indicator,
            )
        };
        check(rc, SQL_HANDLE_STMT, self.handle, "Failed to bind parameter")
    }

    pub fn execute(&self) -> Result<()> {
        let f = fns();
        let rc = unsafe { (f.SQLExecute)(self.handle) };
        check(
            rc,
            SQL_HANDLE_STMT,
            self.handle,
            "Failed to execute prepared statement",
        )
    }

    pub fn reset_params(&self) -> Result<()> {
        let f = fns();
        let rc = unsafe { (f.SQLFreeStmt)(self.handle, SQL_RESET_PARAMS) };
        check(
            rc,
            SQL_HANDLE_STMT,
            self.handle,
            "Failed to reset parameters",
        )
    }
}

impl Drop for PreparedStatement {
    fn drop(&mut self) {
        let f = fns();
        let rc = unsafe { (f.SQLFreeHandle)(SQL_HANDLE_STMT, self.handle) };
        if !succeeded(rc) && !std::thread::panicking() {
            panic!(
                "SQLFreeHandle(STMT) failed: {}",
                extract_diagnostic(SQL_HANDLE_STMT, self.handle)
            );
        }
    }
}

// ============================================================================
// ODBC catalog function helpers
// ============================================================================

/// Execute SQLTables and return the result as a Statement cursor.
pub fn sql_tables(
    conn: &Connection,
    catalog: Option<&str>,
    schema: Option<&str>,
    table: Option<&str>,
    table_type: Option<&str>,
) -> Result<Statement> {
    let f = fns();
    let mut stmt_handle = SQL_NULL_HANDLE;
    let rc = unsafe { (f.SQLAllocHandle)(SQL_HANDLE_STMT, conn.handle(), &mut stmt_handle) };
    check(
        rc,
        SQL_HANDLE_DBC,
        conn.handle(),
        "Failed to allocate statement for SQLTables",
    )?;

    let (cat_cs, cat_len) = str_to_odbc_cstring(catalog)?;
    let (sch_cs, sch_len) = str_to_odbc_cstring(schema)?;
    let (tbl_cs, tbl_len) = str_to_odbc_cstring(table)?;
    let (typ_cs, typ_len) = str_to_odbc_cstring(table_type)?;

    let rc = unsafe {
        (f.SQLTables)(
            stmt_handle,
            cstring_ptr(&cat_cs),
            cat_len,
            cstring_ptr(&sch_cs),
            sch_len,
            cstring_ptr(&tbl_cs),
            tbl_len,
            cstring_ptr(&typ_cs),
            typ_len,
        )
    };
    if !succeeded(rc) {
        let diag = extract_diagnostic(SQL_HANDLE_STMT, stmt_handle);
        unsafe { (f.SQLFreeHandle)(SQL_HANDLE_STMT, stmt_handle) };
        return Err(GgsqlError::ReaderError(format!(
            "SQLTables failed: {}",
            diag
        )));
    }

    Ok(Statement {
        handle: stmt_handle,
    })
}

/// Execute SQLColumns and return the result as a Statement cursor.
pub fn sql_columns(
    conn: &Connection,
    catalog: Option<&str>,
    schema: Option<&str>,
    table: Option<&str>,
    column: Option<&str>,
) -> Result<Statement> {
    let f = fns();
    let mut stmt_handle = SQL_NULL_HANDLE;
    let rc = unsafe { (f.SQLAllocHandle)(SQL_HANDLE_STMT, conn.handle(), &mut stmt_handle) };
    check(
        rc,
        SQL_HANDLE_DBC,
        conn.handle(),
        "Failed to allocate statement for SQLColumns",
    )?;

    let (cat_cs, cat_len) = str_to_odbc_cstring(catalog)?;
    let (sch_cs, sch_len) = str_to_odbc_cstring(schema)?;
    let (tbl_cs, tbl_len) = str_to_odbc_cstring(table)?;
    let (col_cs, col_len) = str_to_odbc_cstring(column)?;

    let rc = unsafe {
        (f.SQLColumns)(
            stmt_handle,
            cstring_ptr(&cat_cs),
            cat_len,
            cstring_ptr(&sch_cs),
            sch_len,
            cstring_ptr(&tbl_cs),
            tbl_len,
            cstring_ptr(&col_cs),
            col_len,
        )
    };
    if !succeeded(rc) {
        let diag = extract_diagnostic(SQL_HANDLE_STMT, stmt_handle);
        unsafe { (f.SQLFreeHandle)(SQL_HANDLE_STMT, stmt_handle) };
        return Err(GgsqlError::ReaderError(format!(
            "SQLColumns failed: {}",
            diag
        )));
    }

    Ok(Statement {
        handle: stmt_handle,
    })
}

fn str_to_odbc_cstring(s: Option<&str>) -> Result<(Option<std::ffi::CString>, SqlSmallInt)> {
    match s {
        Some(s) => {
            let cs = std::ffi::CString::new(s).map_err(|_| {
                GgsqlError::ReaderError("ODBC catalog argument contains null byte".into())
            })?;
            let len = s.len() as SqlSmallInt;
            Ok((Some(cs), len))
        }
        None => Ok((None, 0)),
    }
}

fn cstring_ptr(cs: &Option<std::ffi::CString>) -> *const SqlChar {
    match cs {
        Some(cs) => cs.as_ptr() as *const SqlChar,
        None => std::ptr::null(),
    }
}
