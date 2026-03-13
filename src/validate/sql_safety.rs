//! SQL safety validation to prevent dangerous operations.
//!
//! This module validates SQL queries to block dangerous patterns that could
//! modify data, alter schema, or execute privileged operations.

use crate::{GgsqlError, Result};
use regex::Regex;
use std::sync::LazyLock;

/// Dangerous SQL keywords (case-insensitive)
/// Covers: DDL, DML, permissions, and database-specific dangerous operations
static DANGEROUS_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(DROP|TRUNCATE|ALTER|DELETE|UPDATE|INSERT|GRANT|REVOKE|COPY|ATTACH|DETACH|LOAD|INSTALL|CALL|EXEC|EXECUTE|VACUUM|PRAGMA|IMPORT|EXPORT)\b").unwrap()
});

/// SQL Server dangerous extended procedures (xp_cmdshell allows OS command execution!)
static SQLSERVER_DANGEROUS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(xp_cmdshell|xp_regread|xp_regwrite|xp_dirtree|xp_fileexist|xp_subdirs|sp_OACreate|sp_OAMethod|OPENROWSET|OPENDATASOURCE)\b").unwrap()
});

/// CREATE OR REPLACE pattern
static CREATE_OR_REPLACE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)CREATE\s+OR\s+REPLACE").unwrap()
});

/// Data-modifying CTEs (WITH ... DELETE/UPDATE/INSERT ... RETURNING)
static CTE_DML: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\bWITH\b.*?\b(DELETE|UPDATE|INSERT)\b.*?\bRETURNING\b").unwrap()
});

/// Regex to match line comments (-- ...)
static LINE_COMMENT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"--[^\n]*").unwrap());

/// Regex to match block comments (/* ... */)
static BLOCK_COMMENT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"/\*.*?\*/").unwrap());

/// Regex to match string literals ('...' with escape handling)
static STRING_LITERAL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"'(?:[^'\\]|\\.)*'").unwrap());

/// Validate SQL for dangerous operations.
///
/// Blocks: DROP, DELETE, TRUNCATE, UPDATE, INSERT, ALTER, GRANT, REVOKE,
/// CREATE OR REPLACE, and data-modifying CTEs.
pub fn validate_sql_safety(sql: &str) -> Result<()> {
    if sql.trim().is_empty() {
        return Ok(());
    }

    let preprocessed = preprocess_sql(sql);

    if CREATE_OR_REPLACE.is_match(&preprocessed) {
        return Err(GgsqlError::ValidationError(
            "CREATE OR REPLACE is not allowed for safety reasons".into(),
        ));
    }

    if CTE_DML.is_match(&preprocessed) {
        return Err(GgsqlError::ValidationError(
            "Data-modifying CTEs (WITH ... DELETE/UPDATE/INSERT ... RETURNING) are not allowed"
                .into(),
        ));
    }

    if let Some(caps) = DANGEROUS_PATTERN.captures(&preprocessed) {
        let keyword = caps.get(1).unwrap().as_str().to_uppercase();
        return Err(GgsqlError::ValidationError(format!(
            "{} statements are not allowed for safety reasons",
            keyword
        )));
    }

    // SQL Server dangerous extended procedures
    if let Some(caps) = SQLSERVER_DANGEROUS.captures(&preprocessed) {
        let proc = caps.get(1).unwrap().as_str();
        return Err(GgsqlError::ValidationError(format!(
            "{} is not allowed for safety reasons",
            proc
        )));
    }

    Ok(())
}

/// Remove comments and string literals to avoid false positives.
fn preprocess_sql(sql: &str) -> String {
    let mut result = sql.to_string();
    // Remove line comments (-- ...)
    result = LINE_COMMENT.replace_all(&result, "").to_string();
    // Remove block comments (/* ... */)
    result = BLOCK_COMMENT.replace_all(&result, "").to_string();
    // Replace string literals with placeholder
    result = STRING_LITERAL.replace_all(&result, "'...'").to_string();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // Safe queries
    #[test]
    fn test_safe_select() {
        assert!(validate_sql_safety("SELECT * FROM users").is_ok());
    }

    #[test]
    fn test_safe_compound_statement() {
        // Compound statements ARE allowed in main SQL
        assert!(validate_sql_safety("SELECT 1; SELECT 2").is_ok());
    }

    #[test]
    fn test_keyword_in_string() {
        assert!(validate_sql_safety("SELECT * FROM t WHERE name = 'DROP TABLE'").is_ok());
    }

    #[test]
    fn test_keyword_in_comment() {
        assert!(validate_sql_safety("SELECT * -- DROP TABLE").is_ok());
        assert!(validate_sql_safety("SELECT /* DELETE */ *").is_ok());
    }

    // Dangerous queries
    #[test]
    fn test_drop_blocked() {
        let r = validate_sql_safety("DROP TABLE users");
        assert!(r.is_err());
        assert!(r.unwrap_err().to_string().contains("DROP"));
    }

    #[test]
    fn test_delete_blocked() {
        assert!(validate_sql_safety("DELETE FROM users").is_err());
    }

    #[test]
    fn test_update_blocked() {
        assert!(validate_sql_safety("UPDATE users SET x = 1").is_err());
    }

    #[test]
    fn test_insert_blocked() {
        assert!(validate_sql_safety("INSERT INTO users VALUES (1)").is_err());
    }

    #[test]
    fn test_truncate_blocked() {
        assert!(validate_sql_safety("TRUNCATE TABLE users").is_err());
    }

    #[test]
    fn test_alter_blocked() {
        assert!(validate_sql_safety("ALTER TABLE users ADD x INT").is_err());
    }

    #[test]
    fn test_grant_blocked() {
        assert!(validate_sql_safety("GRANT SELECT ON users TO public").is_err());
    }

    #[test]
    fn test_revoke_blocked() {
        assert!(validate_sql_safety("REVOKE SELECT ON users FROM public").is_err());
    }

    #[test]
    fn test_create_or_replace_blocked() {
        assert!(validate_sql_safety("CREATE OR REPLACE VIEW v AS SELECT 1").is_err());
    }

    #[test]
    fn test_data_modifying_cte_blocked() {
        assert!(
            validate_sql_safety("WITH d AS (DELETE FROM users RETURNING *) SELECT * FROM d")
                .is_err()
        );
    }

    #[test]
    fn test_case_insensitive() {
        assert!(validate_sql_safety("drop table users").is_err());
        assert!(validate_sql_safety("DrOp TaBlE users").is_err());
    }

    // Database-specific dangerous operations
    #[test]
    fn test_copy_blocked() {
        assert!(validate_sql_safety("COPY users TO '/tmp/data.csv'").is_err());
        assert!(validate_sql_safety("COPY users FROM '/tmp/data.csv'").is_err());
    }

    #[test]
    fn test_attach_blocked() {
        assert!(validate_sql_safety("ATTACH DATABASE 'other.db' AS other").is_err());
        assert!(validate_sql_safety("DETACH DATABASE other").is_err());
    }

    #[test]
    fn test_load_install_blocked() {
        assert!(validate_sql_safety("LOAD 'httpfs'").is_err());
        assert!(validate_sql_safety("INSTALL httpfs").is_err());
    }

    #[test]
    fn test_exec_blocked() {
        assert!(validate_sql_safety("EXEC sp_executesql @sql").is_err());
        assert!(validate_sql_safety("EXECUTE sp_help").is_err());
        assert!(validate_sql_safety("CALL my_procedure()").is_err());
    }

    #[test]
    fn test_vacuum_pragma_blocked() {
        assert!(validate_sql_safety("VACUUM").is_err());
        assert!(validate_sql_safety("PRAGMA table_info(users)").is_err());
    }

    #[test]
    fn test_import_export_blocked() {
        assert!(validate_sql_safety("EXPORT DATABASE '/backup'").is_err());
        assert!(validate_sql_safety("IMPORT DATABASE '/backup'").is_err());
    }

    // SQL Server dangerous procedures
    #[test]
    fn test_xp_cmdshell_blocked() {
        // xp_cmdshell allows OS command execution - extremely dangerous!
        assert!(validate_sql_safety("EXEC xp_cmdshell 'whoami'").is_err());
        assert!(validate_sql_safety("xp_cmdshell 'dir'").is_err());
    }

    #[test]
    fn test_xp_registry_blocked() {
        assert!(validate_sql_safety("EXEC xp_regread 'HKEY_LOCAL_MACHINE'").is_err());
        assert!(validate_sql_safety("EXEC xp_regwrite 'HKEY_LOCAL_MACHINE'").is_err());
    }

    #[test]
    fn test_xp_filesystem_blocked() {
        assert!(validate_sql_safety("EXEC xp_dirtree '/etc'").is_err());
        assert!(validate_sql_safety("EXEC xp_fileexist '/etc/passwd'").is_err());
        assert!(validate_sql_safety("EXEC xp_subdirs 'C:\\'").is_err());
    }

    #[test]
    fn test_sp_oa_blocked() {
        // OLE automation can execute arbitrary code
        assert!(validate_sql_safety("EXEC sp_OACreate 'WScript.Shell'").is_err());
        assert!(validate_sql_safety("EXEC sp_OAMethod @obj, 'Run'").is_err());
    }

    #[test]
    fn test_openrowset_blocked() {
        // OPENROWSET can read external files and execute code
        assert!(validate_sql_safety("SELECT * FROM OPENROWSET('SQLNCLI', ...)").is_err());
        assert!(validate_sql_safety("SELECT * FROM OPENDATASOURCE('SQLNCLI', ...)").is_err());
    }
}
