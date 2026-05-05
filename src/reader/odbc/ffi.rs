//! Raw ODBC FFI types, constants, and runtime-loaded function pointers.
//!
//! Loads `libodbc` via `libloading` at runtime.

use std::sync::OnceLock;

// ============================================================================
// Primitive type aliases (must match ODBC spec sizes)
// ============================================================================

pub type SqlChar = u8;
pub type SqlSmallInt = i16;
pub type SqlUSmallInt = u16;
pub type SqlInteger = i32;
pub type SqlUInteger = u32;
pub type SqlLen = isize;
pub type SqlULen = usize;
pub type SqlReturn = i16;
pub type SqlHandle = *mut std::ffi::c_void;
pub type SqlHEnv = SqlHandle;
pub type SqlHDbc = SqlHandle;
pub type SqlHStmt = SqlHandle;
pub type SqlHWnd = SqlHandle;
pub type SqlPointer = *mut std::ffi::c_void;

// ============================================================================
// Return codes
// ============================================================================

pub const SQL_SUCCESS: SqlReturn = 0;
pub const SQL_SUCCESS_WITH_INFO: SqlReturn = 1;
pub const SQL_ERROR: SqlReturn = -1;
pub const SQL_INVALID_HANDLE: SqlReturn = -2;
pub const SQL_NO_DATA: SqlReturn = 100;

#[inline]
pub fn succeeded(rc: SqlReturn) -> bool {
    rc == SQL_SUCCESS || rc == SQL_SUCCESS_WITH_INFO
}

// ============================================================================
// Handle types (for SQLAllocHandle / SQLFreeHandle)
// ============================================================================

pub const SQL_HANDLE_ENV: SqlSmallInt = 1;
pub const SQL_HANDLE_DBC: SqlSmallInt = 2;
pub const SQL_HANDLE_STMT: SqlSmallInt = 3;

pub const SQL_NULL_HANDLE: SqlHandle = std::ptr::null_mut();

// ============================================================================
// Environment attributes
// ============================================================================

pub const SQL_ATTR_ODBC_VERSION: SqlInteger = 200;
pub const SQL_OV_ODBC3_80: SqlInteger = 380;

// ============================================================================
// Connection attributes
// ============================================================================

pub const SQL_ATTR_AUTOCOMMIT: SqlInteger = 102;
pub const SQL_AUTOCOMMIT_ON: SqlUInteger = 1;

// ============================================================================
// Statement attributes
// ============================================================================

pub const SQL_ATTR_ROW_BIND_TYPE: SqlInteger = 5;
pub const SQL_ATTR_ROW_ARRAY_SIZE: SqlInteger = 27;
pub const SQL_ATTR_ROWS_FETCHED_PTR: SqlInteger = 26;
pub const SQL_BIND_BY_COLUMN: SqlULen = 0;

// ============================================================================
// SQLFreeStmt options
// ============================================================================

pub const SQL_CLOSE: SqlUSmallInt = 0;
pub const SQL_UNBIND: SqlUSmallInt = 2;
pub const SQL_RESET_PARAMS: SqlUSmallInt = 3;

// ============================================================================
// SQLDriverConnect options
// ============================================================================

pub const SQL_DRIVER_NOPROMPT: SqlUSmallInt = 0;

// ============================================================================
// SQLEndTran completion types
// ============================================================================

pub const SQL_COMMIT: SqlSmallInt = 0;
pub const SQL_ROLLBACK: SqlSmallInt = 1;

// ============================================================================
// SQL data types (returned by SQLDescribeCol)
// ============================================================================

pub const SQL_UNKNOWN_TYPE: SqlSmallInt = 0;
pub const SQL_CHAR: SqlSmallInt = 1;
pub const SQL_NUMERIC: SqlSmallInt = 2;
pub const SQL_DECIMAL: SqlSmallInt = 3;
pub const SQL_INTEGER: SqlSmallInt = 4;
pub const SQL_SMALLINT: SqlSmallInt = 5;
pub const SQL_FLOAT: SqlSmallInt = 6;
pub const SQL_REAL: SqlSmallInt = 7;
pub const SQL_DOUBLE: SqlSmallInt = 8;
pub const SQL_VARCHAR: SqlSmallInt = 12;
pub const SQL_TYPE_DATE: SqlSmallInt = 91;
pub const SQL_TYPE_TIME: SqlSmallInt = 92;
pub const SQL_TYPE_TIMESTAMP: SqlSmallInt = 93;
pub const SQL_BIGINT: SqlSmallInt = -5;
pub const SQL_TINYINT: SqlSmallInt = -6;
pub const SQL_BIT: SqlSmallInt = -7;
pub const SQL_WCHAR: SqlSmallInt = -8;
pub const SQL_WVARCHAR: SqlSmallInt = -9;
pub const SQL_LONGVARCHAR: SqlSmallInt = -1;
pub const SQL_WLONGVARCHAR: SqlSmallInt = -10;

// ============================================================================
// C data types (for SQLBindCol / SQLBindParameter / SQLGetData)
// ============================================================================

pub const SQL_C_CHAR: SqlSmallInt = 1;
pub const SQL_C_SLONG: SqlSmallInt = -16;
pub const SQL_C_SSHORT: SqlSmallInt = -15;
pub const SQL_C_STINYINT: SqlSmallInt = -26;
pub const SQL_C_SBIGINT: SqlSmallInt = -25;
pub const SQL_C_FLOAT: SqlSmallInt = 7;
pub const SQL_C_DOUBLE: SqlSmallInt = 8;
pub const SQL_C_BIT: SqlSmallInt = -7;
pub const SQL_C_TYPE_DATE: SqlSmallInt = 91;
pub const SQL_C_TYPE_TIME: SqlSmallInt = 92;
pub const SQL_C_TYPE_TIMESTAMP: SqlSmallInt = 93;

// ============================================================================
// Indicator / length constants
// ============================================================================

pub const SQL_NULL_DATA: SqlLen = -1;
pub const SQL_NTS: SqlLen = -3;

// ============================================================================
// SQLGetInfo info types
// ============================================================================

pub const SQL_DBMS_NAME: SqlUSmallInt = 17;

// ============================================================================
// SQLBindParameter input/output type
// ============================================================================

pub const SQL_PARAM_INPUT: SqlSmallInt = 1;

// ============================================================================
// Nullable constants (returned by SQLDescribeCol)
// ============================================================================

pub const SQL_NO_NULLS: SqlSmallInt = 0;
pub const SQL_NULLABLE: SqlSmallInt = 1;

// ============================================================================
// Date/time structs (matching ODBC C structs)
// ============================================================================

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SqlDateStruct {
    pub year: SqlSmallInt,
    pub month: SqlUSmallInt,
    pub day: SqlUSmallInt,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SqlTimeStruct {
    pub hour: SqlUSmallInt,
    pub minute: SqlUSmallInt,
    pub second: SqlUSmallInt,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SqlTimestampStruct {
    pub year: SqlSmallInt,
    pub month: SqlUSmallInt,
    pub day: SqlUSmallInt,
    pub hour: SqlUSmallInt,
    pub minute: SqlUSmallInt,
    pub second: SqlUSmallInt,
    pub fraction: u32,
}

// ============================================================================
// Runtime-loaded function table
// ============================================================================

macro_rules! odbc_functions {
    ($(
        $(#[$meta:meta])*
        fn $name:ident( $($arg:ident : $argty:ty),* $(,)? ) -> SqlReturn;
    )*) => {
        #[allow(non_snake_case)]
        pub(crate) struct OdbcFunctions {
            $( pub $name: unsafe extern "system" fn( $($argty),* ) -> SqlReturn, )*
        }

        unsafe fn load_functions(lib: &libloading::Library) -> Result<OdbcFunctions, String> {
            unsafe {
                Ok(OdbcFunctions {
                    $( $name: *lib.get(concat!(stringify!($name), "\0").as_bytes())
                        .map_err(|e| format!("{}: {}", stringify!($name), e))?, )*
                })
            }
        }
    };
}

odbc_functions! {
    fn SQLAllocHandle(handle_type: SqlSmallInt, input_handle: SqlHandle, output_handle: *mut SqlHandle) -> SqlReturn;
    fn SQLFreeHandle(handle_type: SqlSmallInt, handle: SqlHandle) -> SqlReturn;
    fn SQLSetEnvAttr(env: SqlHEnv, attribute: SqlInteger, value: SqlPointer, string_length: SqlInteger) -> SqlReturn;
    fn SQLDriverConnect(
        dbc: SqlHDbc, hwnd: SqlHWnd,
        in_conn_str: *const SqlChar, in_len: SqlSmallInt,
        out_conn_str: *mut SqlChar, out_max: SqlSmallInt, out_len: *mut SqlSmallInt,
        driver_completion: SqlUSmallInt
    ) -> SqlReturn;
    fn SQLDisconnect(dbc: SqlHDbc) -> SqlReturn;
    fn SQLExecDirect(stmt: SqlHStmt, text: *const SqlChar, text_length: SqlInteger) -> SqlReturn;
    fn SQLPrepare(stmt: SqlHStmt, text: *const SqlChar, text_length: SqlInteger) -> SqlReturn;
    fn SQLExecute(stmt: SqlHStmt) -> SqlReturn;
    fn SQLNumResultCols(stmt: SqlHStmt, column_count: *mut SqlSmallInt) -> SqlReturn;
    fn SQLDescribeCol(
        stmt: SqlHStmt, col_number: SqlUSmallInt,
        col_name: *mut SqlChar, buf_len: SqlSmallInt, name_len: *mut SqlSmallInt,
        data_type: *mut SqlSmallInt, col_size: *mut SqlULen,
        decimal_digits: *mut SqlSmallInt, nullable: *mut SqlSmallInt
    ) -> SqlReturn;
    fn SQLBindCol(
        stmt: SqlHStmt, col_number: SqlUSmallInt, target_type: SqlSmallInt,
        target_value: SqlPointer, buffer_length: SqlLen, indicator: *mut SqlLen
    ) -> SqlReturn;
    fn SQLFetch(stmt: SqlHStmt) -> SqlReturn;
    fn SQLBindParameter(
        stmt: SqlHStmt, param_number: SqlUSmallInt, input_output_type: SqlSmallInt,
        value_type: SqlSmallInt, parameter_type: SqlSmallInt,
        column_size: SqlULen, decimal_digits: SqlSmallInt,
        parameter_value: SqlPointer, buffer_length: SqlLen,
        str_len_or_ind: *mut SqlLen
    ) -> SqlReturn;
    fn SQLGetDiagRec(
        handle_type: SqlSmallInt, handle: SqlHandle, rec_number: SqlSmallInt,
        sql_state: *mut SqlChar, native_error: *mut SqlInteger,
        message_text: *mut SqlChar, buffer_length: SqlSmallInt,
        text_length: *mut SqlSmallInt
    ) -> SqlReturn;
    fn SQLFreeStmt(stmt: SqlHStmt, option: SqlUSmallInt) -> SqlReturn;
    fn SQLRowCount(stmt: SqlHStmt, row_count: *mut SqlLen) -> SqlReturn;
    fn SQLSetStmtAttr(stmt: SqlHStmt, attribute: SqlInteger, value: SqlPointer, string_length: SqlInteger) -> SqlReturn;
    fn SQLGetInfo(
        dbc: SqlHDbc, info_type: SqlUSmallInt,
        info_value: SqlPointer, buffer_length: SqlSmallInt,
        string_length: *mut SqlSmallInt
    ) -> SqlReturn;
    fn SQLTables(
        stmt: SqlHStmt,
        catalog: *const SqlChar, catalog_len: SqlSmallInt,
        schema: *const SqlChar, schema_len: SqlSmallInt,
        table: *const SqlChar, table_len: SqlSmallInt,
        table_type: *const SqlChar, table_type_len: SqlSmallInt
    ) -> SqlReturn;
    fn SQLColumns(
        stmt: SqlHStmt,
        catalog: *const SqlChar, catalog_len: SqlSmallInt,
        schema: *const SqlChar, schema_len: SqlSmallInt,
        table: *const SqlChar, table_len: SqlSmallInt,
        column: *const SqlChar, column_len: SqlSmallInt
    ) -> SqlReturn;
    fn SQLSetConnectAttr(dbc: SqlHDbc, attribute: SqlInteger, value: SqlPointer, string_length: SqlInteger) -> SqlReturn;
    fn SQLEndTran(handle_type: SqlSmallInt, handle: SqlHandle, completion_type: SqlSmallInt) -> SqlReturn;
}

struct OdbcLibrary {
    _lib: libloading::Library,
    pub fns: OdbcFunctions,
}

// Safety: libloading::Library is Send+Sync. Function pointers are valid for
// the library's lifetime, and ODBC thread safety is the caller's responsibility.
unsafe impl Send for OdbcLibrary {}
unsafe impl Sync for OdbcLibrary {}

static ODBC: OnceLock<Result<OdbcLibrary, String>> = OnceLock::new();

fn load_odbc() -> Result<OdbcLibrary, String> {
    // Check env var first
    if let Ok(path) = std::env::var("GGSQL_ODBC_LIBRARY") {
        match unsafe { libloading::Library::new(&path) } {
            Ok(lib) => {
                let fns = unsafe { load_functions(&lib)? };
                return Ok(OdbcLibrary { _lib: lib, fns });
            }
            Err(e) => {
                return Err(format!(
                    "GGSQL_ODBC_LIBRARY={} could not be loaded: {}",
                    path, e
                ));
            }
        }
    }

    let mut names: Vec<String> = Vec::new();

    #[cfg(target_os = "linux")]
    {
        names.push("libodbc.so.2".into());
        names.push("libodbc.so".into());
    }
    #[cfg(target_os = "macos")]
    {
        names.push("libodbc.2.dylib".into());
        names.push("libodbc.dylib".into());
        // Homebrew on Apple Silicon
        names.push("/opt/homebrew/lib/libodbc.2.dylib".into());
        // Homebrew on Intel
        names.push("/usr/local/lib/libodbc.2.dylib".into());
    }
    #[cfg(target_os = "windows")]
    {
        names.push("odbc32.dll".into());
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        names.push("libodbc.so.2".into());
        names.push("libodbc.so".into());
    }

    let mut errors = Vec::new();
    for name in &names {
        match unsafe { libloading::Library::new(name) } {
            Ok(lib) => {
                let fns = unsafe { load_functions(&lib)? };
                return Ok(OdbcLibrary { _lib: lib, fns });
            }
            Err(e) => errors.push(format!("  {}: {}", name, e)),
        }
    }

    Err(format!(
        "ODBC driver manager not found. Install unixODBC for your platform:\n\
         \n\
         \x20 macOS:   brew install unixodbc\n\
         \x20 Debian:  sudo apt install unixodbc\n\
         \x20 Fedora:  sudo dnf install unixODBC\n\
         \x20 RHEL:    sudo yum install unixODBC\n\
         \n\
         Or set GGSQL_ODBC_LIBRARY to the path of your ODBC driver manager.\n\
         \n\
         Tried:\n{}",
        errors.join("\n")
    ))
}

/// Pre-load the ODBC driver manager. Returns Ok(()) if available.
pub fn try_load() -> Result<(), String> {
    match ODBC.get_or_init(load_odbc) {
        Ok(_) => Ok(()),
        Err(e) => Err(e.clone()),
    }
}

/// Get the loaded ODBC function table. Panics if not loaded.
pub(crate) fn fns() -> &'static OdbcFunctions {
    match ODBC.get_or_init(load_odbc) {
        Ok(lib) => &lib.fns,
        Err(e) => panic!(
            "ODBC function called but driver manager is not available: {}",
            e
        ),
    }
}
