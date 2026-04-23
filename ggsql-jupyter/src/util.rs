use arrow::array::ArrayRef;
use ggsql::DataFrame;

/// Find a DataFrame column by name, trying multiple names and falling back to
/// case-insensitive matching. This handles ODBC drivers that return uppercase
/// column names (e.g. `TABLE_NAME` instead of `table_name`).
pub fn find_column<'a>(df: &'a DataFrame, names: &[&str]) -> Result<&'a ArrayRef, String> {
    // Try exact match first
    for name in names {
        if let Ok(col) = df.column(name) {
            return Ok(col);
        }
    }
    // Fall back to case-insensitive match
    let col_names = df.get_column_names();
    for name in names {
        let lower = name.to_lowercase();
        for cn in &col_names {
            if cn.to_lowercase() == lower {
                return df.column(cn).map_err(|e| e.to_string());
            }
        }
    }
    Err(format!("Missing column (tried: {:?})", names))
}
