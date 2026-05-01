//! Snowflake Workbench credential detection and connection resolution.

pub(super) fn is_snowflake(conn_str: &str) -> bool {
    conn_str.to_lowercase().contains("driver=snowflake")
}

pub(super) fn has_token(conn_str: &str) -> bool {
    conn_str.to_lowercase().contains("token=")
}

fn home_dir() -> Option<std::path::PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var("USERPROFILE")
            .ok()
            .map(std::path::PathBuf::from)
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::env::var("HOME").ok().map(std::path::PathBuf::from)
    }
}

/// Find the Snowflake connections.toml file, checking standard locations.
fn find_snowflake_connections_toml() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;

    if let Ok(snowflake_home) = std::env::var("SNOWFLAKE_HOME") {
        let p = PathBuf::from(&snowflake_home).join("connections.toml");
        if p.exists() {
            return Some(p);
        }
    }

    if let Some(home) = home_dir() {
        let p = home.join(".snowflake").join("connections.toml");
        if p.exists() {
            return Some(p);
        }
    }

    if let Some(home) = home_dir() {
        #[cfg(target_os = "macos")]
        {
            let p = home.join("Library/Application Support/snowflake/connections.toml");
            if p.exists() {
                return Some(p);
            }
        }

        #[cfg(target_os = "linux")]
        {
            let xdg = std::env::var("XDG_CONFIG_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|_| home.join(".config"));
            let p = xdg.join("snowflake").join("connections.toml");
            if p.exists() {
                return Some(p);
            }
        }

        #[cfg(target_os = "windows")]
        {
            let p = home.join("AppData/Local/snowflake/connections.toml");
            if p.exists() {
                return Some(p);
            }
        }
    }

    None
}

/// Resolve a `ConnectionName=<name>` parameter in a Snowflake ODBC connection
/// string by reading the named entry from `~/.snowflake/connections.toml` and
/// building a full ODBC connection string from it.
pub(super) fn resolve_connection_name(conn_str: &str) -> Option<String> {
    let lower = conn_str.to_lowercase();
    let cn_key = "connectionname=";
    let cn_start = lower.find(cn_key)?;
    let value_start = cn_start + cn_key.len();

    let rest = &conn_str[value_start..];
    let value_end = rest.find(';').unwrap_or(rest.len());
    let connection_name = rest[..value_end].trim();

    if connection_name.is_empty() {
        return None;
    }

    let toml_path = find_snowflake_connections_toml()?;
    let content = std::fs::read_to_string(&toml_path).ok()?;
    let doc = content.parse::<toml_edit::DocumentMut>().ok()?;

    let entry = doc.get(connection_name)?;
    if !entry.is_table() && !entry.is_inline_table() {
        return None;
    }

    let get_str = |key: &str| -> Option<String> { entry.get(key)?.as_str().map(|s| s.to_string()) };

    let account = get_str("account")?;
    let mut parts = vec![
        "Driver=Snowflake".to_string(),
        format!("Server={}.snowflakecomputing.com", account),
    ];

    if let Some(user) = get_str("user") {
        parts.push(format!("UID={}", user));
    }
    if let Some(password) = get_str("password") {
        parts.push(format!("PWD={}", password));
    }
    if let Some(authenticator) = get_str("authenticator") {
        parts.push(format!("Authenticator={}", authenticator));
    }
    if let Some(token) = get_str("token") {
        parts.push(format!("Token={}", token));
    }
    if let Some(warehouse) = get_str("warehouse") {
        parts.push(format!("Warehouse={}", warehouse));
    }
    if let Some(database) = get_str("database") {
        parts.push(format!("Database={}", database));
    }
    if let Some(schema) = get_str("schema") {
        parts.push(format!("Schema={}", schema));
    }
    if let Some(role) = get_str("role") {
        parts.push(format!("Role={}", role));
    }

    Some(parts.join(";"))
}

/// Detect Posit Workbench Snowflake OAuth token.
pub(super) fn detect_workbench_token() -> Option<String> {
    let snowflake_home = std::env::var("SNOWFLAKE_HOME").ok()?;

    if !snowflake_home.contains("posit-workbench") {
        return None;
    }

    let toml_path = std::path::Path::new(&snowflake_home).join("connections.toml");
    let content = std::fs::read_to_string(&toml_path).ok()?;

    let doc = content.parse::<toml_edit::DocumentMut>().ok()?;
    let token = doc.get("workbench")?.get("token")?.as_str()?.to_string();

    if token.is_empty() {
        None
    } else {
        Some(token)
    }
}

/// Inject OAuth token into a Snowflake ODBC connection string.
pub(super) fn inject_snowflake_token(conn_str: &str, token: &str) -> String {
    let mut result = conn_str.trim_end_matches(';').to_string();
    result.push_str(";Authenticator=oauth;Token=");
    result.push_str(token);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_snowflake() {
        assert!(is_snowflake(
            "Driver=Snowflake;Server=foo.snowflakecomputing.com"
        ));
        assert!(!is_snowflake("Driver={PostgreSQL};Server=localhost"));
    }

    #[test]
    fn test_has_token() {
        assert!(has_token("Driver=Snowflake;Token=abc123"));
        assert!(!has_token("Driver=Snowflake;Server=foo"));
    }

    #[test]
    fn test_inject_snowflake_token() {
        let result = inject_snowflake_token(
            "Driver=Snowflake;Server=foo.snowflakecomputing.com",
            "mytoken",
        );
        assert!(result.contains("Authenticator=oauth"));
        assert!(result.contains("Token=mytoken"));
    }

    #[test]
    fn test_resolve_connection_name_with_toml() {
        use std::io::Write;

        let dir = tempfile::tempdir().unwrap();
        let toml_path = dir.path().join("connections.toml");
        let mut f = std::fs::File::create(&toml_path).unwrap();
        writeln!(
            f,
            r#"
default_connection_name = "myconn"

[myconn]
account = "myaccount"
user = "myuser"
password = "mypass"
warehouse = "mywh"
database = "mydb"
schema = "public"
role = "myrole"

[other]
account = "otheraccount"
"#
        )
        .unwrap();

        std::env::set_var("SNOWFLAKE_HOME", dir.path());

        let result = resolve_connection_name("Driver=Snowflake;ConnectionName=myconn");
        assert!(result.is_some());
        let conn = result.unwrap();
        assert!(conn.contains("Driver=Snowflake"));
        assert!(conn.contains("Server=myaccount.snowflakecomputing.com"));
        assert!(conn.contains("UID=myuser"));
        assert!(conn.contains("PWD=mypass"));
        assert!(conn.contains("Warehouse=mywh"));
        assert!(conn.contains("Database=mydb"));
        assert!(conn.contains("Schema=public"));
        assert!(conn.contains("Role=myrole"));

        let result2 = resolve_connection_name("Driver=Snowflake;ConnectionName=other");
        assert!(result2.is_some());
        let conn2 = result2.unwrap();
        assert!(conn2.contains("Server=otheraccount.snowflakecomputing.com"));
        assert!(!conn2.contains("UID="));

        let result3 = resolve_connection_name("Driver=Snowflake;ConnectionName=nonexistent");
        assert!(result3.is_none());

        let result4 = resolve_connection_name("Driver=Snowflake;Server=foo");
        assert!(result4.is_none());

        std::env::remove_var("SNOWFLAKE_HOME");
    }
}
