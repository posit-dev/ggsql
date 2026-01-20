/*!
ggsql REST API Server

Provides HTTP endpoints for executing ggsql queries and returning visualization outputs.

## Usage

```bash
ggsql-rest --host 127.0.0.1 --port 3000
```

## Endpoints

- `POST /api/v1/query` - Execute a ggsql query
- `POST /api/v1/parse` - Parse a ggsql query (debugging)
- `GET /api/v1/health` - Health check
- `GET /api/v1/version` - Version information
*/

use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use ggsql::{parser, GgsqlError, VERSION};

#[cfg(feature = "duckdb")]
use ggsql::execute::prepare_data_with_executor;
#[cfg(feature = "duckdb")]
use ggsql::reader::{DuckDBReader, Reader};

#[cfg(feature = "vegalite")]
use ggsql::writer::{VegaLiteWriter, Writer};

/// CLI arguments for the REST API server
#[derive(Parser)]
#[command(name = "ggsql-rest")]
#[command(about = "ggsql REST API Server")]
#[command(version = VERSION)]
struct Cli {
    /// Host address to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port number to bind to
    #[arg(long, default_value = "3334")]
    port: u16,

    /// CORS allowed origins (comma-separated)
    #[arg(long, default_value = "*")]
    cors_origin: String,

    /// Load sample data into in-memory database
    #[arg(long, default_value = "false")]
    load_sample_data: bool,

    /// Load data from file(s) into in-memory database
    /// Supports: CSV, Parquet, JSON
    /// Example: --load-data data.csv --load-data other.parquet
    #[arg(long = "load-data")]
    load_data_files: Vec<String>,
}

/// Shared application state
#[derive(Clone)]
struct AppState {
    /// Pre-initialized DuckDB reader with loaded data
    /// Wrapped in Arc<Mutex> since DuckDB Connection is not Sync
    #[cfg(feature = "duckdb")]
    reader: Option<std::sync::Arc<std::sync::Mutex<DuckDBReader>>>,
}

// ============================================================================
// Request/Response Types
// ============================================================================

/// Request body for /api/v1/query endpoint
#[derive(Debug, Deserialize)]
struct QueryRequest {
    /// ggsql query to execute
    query: String,
    /// Data source connection string (optional, default: duckdb://memory)
    #[serde(default = "default_reader")]
    reader: String,
    /// Output writer format (optional, default: vegalite)
    #[serde(default = "default_writer")]
    writer: String,
}

fn default_reader() -> String {
    "duckdb://memory".to_string()
}

fn default_writer() -> String {
    "vegalite".to_string()
}

/// Request body for /api/v1/parse endpoint
#[derive(Debug, Deserialize)]
struct ParseRequest {
    /// ggsql query to parse
    query: String,
}

/// Successful API response
#[derive(Debug, Serialize)]
struct ApiSuccess<T> {
    status: String,
    data: T,
}

/// Error API response
#[derive(Debug, Serialize)]
struct ApiError {
    status: String,
    error: ErrorDetails,
}

#[derive(Debug, Serialize)]
struct ErrorDetails {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
}

/// Query execution result data
#[derive(Debug, Serialize)]
struct QueryResult {
    /// The visualization specification (Vega-Lite JSON, etc.)
    spec: serde_json::Value,
    /// Metadata about the query execution
    metadata: QueryMetadata,
}

#[derive(Debug, Serialize)]
struct QueryMetadata {
    rows: usize,
    columns: Vec<String>,
    global_mappings: String,
    layers: usize,
}

/// Parse result data
#[derive(Debug, Serialize)]
struct ParseResult {
    sql_portion: String,
    viz_portion: String,
    specs: Vec<serde_json::Value>,
}

/// Health check response
#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    version: String,
}

/// Version response
#[derive(Debug, Serialize)]
struct VersionResponse {
    version: String,
    features: Vec<String>,
}

// ============================================================================
// Error Handling
// ============================================================================

/// Custom error type for API responses
struct ApiErrorResponse {
    status: StatusCode,
    error: ApiError,
}

impl IntoResponse for ApiErrorResponse {
    fn into_response(self) -> Response {
        let json = Json(self.error);
        (self.status, json).into_response()
    }
}

impl From<GgsqlError> for ApiErrorResponse {
    fn from(err: GgsqlError) -> Self {
        let (status, error_type) = match &err {
            GgsqlError::ParseError(_) => (StatusCode::BAD_REQUEST, "ParseError"),
            GgsqlError::ValidationError(_) => (StatusCode::BAD_REQUEST, "ValidationError"),
            GgsqlError::ReaderError(_) => (StatusCode::BAD_REQUEST, "ReaderError"),
            GgsqlError::WriterError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "WriterError"),
            GgsqlError::InternalError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "InternalError"),
        };

        ApiErrorResponse {
            status,
            error: ApiError {
                status: "error".to_string(),
                error: ErrorDetails {
                    message: err.to_string(),
                    error_type: error_type.to_string(),
                },
            },
        }
    }
}

impl From<String> for ApiErrorResponse {
    fn from(msg: String) -> Self {
        ApiErrorResponse {
            status: StatusCode::BAD_REQUEST,
            error: ApiError {
                status: "error".to_string(),
                error: ErrorDetails {
                    message: msg,
                    error_type: "BadRequest".to_string(),
                },
            },
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

#[cfg(feature = "duckdb")]
fn load_data_files(reader: &DuckDBReader, files: &[String]) -> Result<(), GgsqlError> {
    use duckdb::params;
    use std::path::Path;

    let conn = reader.connection();

    for file_path in files {
        let path = Path::new(file_path);

        if !path.exists() {
            return Err(GgsqlError::ReaderError(format!(
                "File not found: {}",
                file_path
            )));
        }

        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        // Derive table name from filename (without extension)
        let table_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("data")
            .replace('-', "_")
            .replace(' ', "_");

        info!("Loading {} into table '{}'", file_path, table_name);

        match extension.as_str() {
            "csv" => {
                // DuckDB can read CSV directly
                let sql = format!(
                    "CREATE TABLE {} AS SELECT * FROM read_csv_auto('{}')",
                    table_name, file_path
                );
                conn.execute(&sql, params![]).map_err(|e| {
                    GgsqlError::ReaderError(format!("Failed to load CSV {}: {}", file_path, e))
                })?;
            }
            "parquet" => {
                // DuckDB can read Parquet directly
                let sql = format!(
                    "CREATE TABLE {} AS SELECT * FROM read_parquet('{}')",
                    table_name, file_path
                );
                conn.execute(&sql, params![]).map_err(|e| {
                    GgsqlError::ReaderError(format!("Failed to load Parquet {}: {}", file_path, e))
                })?;
            }
            "json" | "jsonl" | "ndjson" => {
                // DuckDB can read JSON directly
                let sql = format!(
                    "CREATE TABLE {} AS SELECT * FROM read_json_auto('{}')",
                    table_name, file_path
                );
                conn.execute(&sql, params![]).map_err(|e| {
                    GgsqlError::ReaderError(format!("Failed to load JSON {}: {}", file_path, e))
                })?;
            }
            _ => {
                return Err(GgsqlError::ReaderError(format!(
                    "Unsupported file format: {} (supported: csv, parquet, json, jsonl, ndjson)",
                    extension
                )));
            }
        }

        info!(
            "Successfully loaded {} as table '{}'",
            file_path, table_name
        );
    }

    Ok(())
}

#[cfg(feature = "duckdb")]
fn load_sample_data(reader: &DuckDBReader) -> Result<(), GgsqlError> {
    use duckdb::params;

    let conn = reader.connection();

    // Create sample products table
    conn.execute(
        "CREATE TABLE products (
            product_id INTEGER,
            product_name VARCHAR,
            category VARCHAR,
            price DECIMAL(10,2)
        )",
        params![],
    )
    .map_err(|e| GgsqlError::ReaderError(format!("Failed to create products table: {}", e)))?;

    conn.execute(
        "INSERT INTO products VALUES
            (1, 'Laptop', 'Electronics', 999.99),
            (2, 'Mouse', 'Electronics', 25.50),
            (3, 'Keyboard', 'Electronics', 75.00),
            (4, 'Desk', 'Furniture', 299.99),
            (5, 'Chair', 'Furniture', 199.99),
            (6, 'Monitor', 'Electronics', 349.99),
            (7, 'Lamp', 'Furniture', 45.00)",
        params![],
    )
    .map_err(|e| GgsqlError::ReaderError(format!("Failed to insert products: {}", e)))?;

    // Create sample sales table with more temporal data
    conn.execute(
        "CREATE TABLE sales (
            sale_id INTEGER,
            product_id INTEGER,
            quantity INTEGER,
            sale_date DATE,
            region VARCHAR
        )",
        params![],
    )
    .map_err(|e| GgsqlError::ReaderError(format!("Failed to create sales table: {}", e)))?;

    conn.execute(
        "INSERT INTO sales VALUES
            -- January 2024
            (1, 1, 2, '2024-01-05', 'US'),
            (2, 2, 5, '2024-01-05', 'EU'),
            (3, 3, 3, '2024-01-05', 'APAC'),
            (4, 1, 3, '2024-01-12', 'US'),
            (5, 2, 4, '2024-01-12', 'EU'),
            (6, 3, 2, '2024-01-12', 'APAC'),
            (7, 4, 2, '2024-01-19', 'US'),
            (8, 5, 1, '2024-01-19', 'EU'),
            (9, 6, 2, '2024-01-19', 'APAC'),
            (10, 1, 4, '2024-01-26', 'US'),
            (11, 2, 3, '2024-01-26', 'EU'),
            (12, 3, 5, '2024-01-26', 'APAC'),
            -- February 2024
            (13, 4, 3, '2024-02-02', 'US'),
            (14, 5, 2, '2024-02-02', 'EU'),
            (15, 6, 1, '2024-02-02', 'APAC'),
            (16, 1, 5, '2024-02-09', 'US'),
            (17, 2, 6, '2024-02-09', 'EU'),
            (18, 3, 4, '2024-02-09', 'APAC'),
            (19, 7, 2, '2024-02-16', 'US'),
            (20, 4, 3, '2024-02-16', 'EU'),
            (21, 5, 2, '2024-02-16', 'APAC'),
            (22, 1, 6, '2024-02-23', 'US'),
            (23, 2, 5, '2024-02-23', 'EU'),
            (24, 6, 3, '2024-02-23', 'APAC'),
            -- March 2024
            (25, 3, 4, '2024-03-01', 'US'),
            (26, 4, 5, '2024-03-01', 'EU'),
            (27, 5, 3, '2024-03-01', 'APAC'),
            (28, 1, 7, '2024-03-08', 'US'),
            (29, 2, 6, '2024-03-08', 'EU'),
            (30, 3, 5, '2024-03-08', 'APAC'),
            (31, 6, 2, '2024-03-15', 'US'),
            (32, 7, 3, '2024-03-15', 'EU'),
            (33, 4, 4, '2024-03-15', 'APAC'),
            (34, 1, 8, '2024-03-22', 'US'),
            (35, 2, 7, '2024-03-22', 'EU'),
            (36, 5, 6, '2024-03-22', 'APAC')",
        params![],
    )
    .map_err(|e| GgsqlError::ReaderError(format!("Failed to insert sales: {}", e)))?;

    // Create sample employees table
    conn.execute(
        "CREATE TABLE employees (
            employee_id INTEGER,
            employee_name VARCHAR,
            department VARCHAR,
            salary INTEGER,
            hire_date DATE
        )",
        params![],
    )
    .map_err(|e| GgsqlError::ReaderError(format!("Failed to create employees table: {}", e)))?;

    conn.execute(
        "INSERT INTO employees VALUES
            (1, 'Alice Johnson', 'Engineering', 95000, '2022-01-15'),
            (2, 'Bob Smith', 'Engineering', 85000, '2022-03-20'),
            (3, 'Carol Williams', 'Sales', 70000, '2022-06-10'),
            (4, 'David Brown', 'Sales', 75000, '2023-01-05'),
            (5, 'Eve Davis', 'Marketing', 65000, '2023-03-15'),
            (6, 'Frank Miller', 'Engineering', 105000, '2021-09-01')",
        params![],
    )
    .map_err(|e| GgsqlError::ReaderError(format!("Failed to insert employees: {}", e)))?;

    Ok(())
}

// ============================================================================
// Handler Functions
// ============================================================================

/// POST /api/v1/query - Execute a ggsql query
async fn query_handler(
    State(state): State<AppState>,
    Json(request): Json<QueryRequest>,
) -> Result<Json<ApiSuccess<QueryResult>>, ApiErrorResponse> {
    info!("Executing query: {} chars", request.query.len());
    info!("Reader: {}, Writer: {}", request.reader, request.writer);

    #[cfg(feature = "duckdb")]
    if request.reader.starts_with("duckdb://") {
        // Create query executor that handles shared state vs new reader
        let execute_query = |sql: &str| -> Result<ggsql::DataFrame, GgsqlError> {
            if request.reader == "duckdb://memory" && state.reader.is_some() {
                let reader_mutex = state.reader.as_ref().unwrap();
                let reader = reader_mutex.lock().map_err(|e| {
                    GgsqlError::InternalError(format!("Failed to lock reader: {}", e))
                })?;
                reader.execute(sql)
            } else {
                let reader = DuckDBReader::from_connection_string(&request.reader)?;
                reader.execute(sql)
            }
        };

        // Prepare data using shared execution logic
        let prepared = prepare_data_with_executor(&request.query, execute_query)?;

        // Get metadata from available data
        let (rows, columns) = if let Some(df) = prepared.data.get("__global__") {
            let (r, _) = df.shape();
            let cols: Vec<String> = df
                .get_column_names()
                .iter()
                .map(|s| s.to_string())
                .collect();
            (r, cols)
        } else {
            // Use first available data for metadata
            let df = prepared.data.values().next().unwrap();
            let (r, _) = df.shape();
            let cols: Vec<String> = df
                .get_column_names()
                .iter()
                .map(|s| s.to_string())
                .collect();
            (r, cols)
        };

        let first_spec = &prepared.specs[0];

        // Generate visualization output using writer
        #[cfg(feature = "vegalite")]
        if request.writer == "vegalite" {
            let writer = VegaLiteWriter::new();
            let json_output = writer.write(first_spec, &prepared.data)?;
            let spec_value: serde_json::Value = serde_json::from_str(&json_output)
                .map_err(|e| GgsqlError::WriterError(format!("Failed to parse JSON: {}", e)))?;

            let result = QueryResult {
                spec: spec_value,
                metadata: QueryMetadata {
                    rows,
                    columns,
                    global_mappings: format!("{:?}", first_spec.global_mappings),
                    layers: first_spec.layers.len(),
                },
            };

            return Ok(Json(ApiSuccess {
                status: "success".to_string(),
                data: result,
            }));
        }

        #[cfg(not(feature = "vegalite"))]
        return Err(ApiErrorResponse::from(
            "VegaLite writer not available".to_string(),
        ));
    }

    #[cfg(not(feature = "duckdb"))]
    return Err(ApiErrorResponse::from(
        "DuckDB reader not available".to_string(),
    ));

    #[cfg(feature = "duckdb")]
    Err(ApiErrorResponse::from(format!(
        "Unsupported reader: {}",
        request.reader
    )))
}

/// POST /api/v1/parse - Parse a ggsql query
async fn parse_handler(
    Json(request): Json<ParseRequest>,
) -> Result<Json<ApiSuccess<ParseResult>>, ApiErrorResponse> {
    info!("Parsing query: {} chars", request.query.len());

    // Split query
    let (sql_part, viz_part) = parser::split_query(&request.query)?;

    // Parse ggsql portion
    let specs = parser::parse_query(&request.query)?;

    // Convert specs to JSON
    let specs_json: Vec<serde_json::Value> = specs
        .iter()
        .map(|spec| serde_json::to_value(spec).unwrap_or(serde_json::Value::Null))
        .collect();

    let result = ParseResult {
        sql_portion: sql_part,
        viz_portion: viz_part,
        specs: specs_json,
    };

    Ok(Json(ApiSuccess {
        status: "success".to_string(),
        data: result,
    }))
}

/// GET /api/v1/health - Health check
async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: VERSION.to_string(),
    })
}

/// GET /api/v1/version - Version information
async fn version_handler() -> Json<VersionResponse> {
    let mut features = Vec::new();

    #[cfg(feature = "duckdb")]
    features.push("duckdb".to_string());

    #[cfg(feature = "vegalite")]
    features.push("vegalite".to_string());

    #[cfg(feature = "sqlite")]
    features.push("sqlite".to_string());

    #[cfg(feature = "postgres")]
    features.push("postgres".to_string());

    Json(VersionResponse {
        version: VERSION.to_string(),
        features,
    })
}

/// Root handler
async fn root_handler() -> &'static str {
    "ggsql REST API Server - See /api/v1/health for status"
}

// ============================================================================
// Main Server
// ============================================================================

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ggsql_rest=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Parse CLI arguments
    let cli = Cli::parse();

    // Initialize DuckDB reader with data if requested
    #[cfg(feature = "duckdb")]
    let reader = if cli.load_sample_data || !cli.load_data_files.is_empty() {
        info!("Initializing in-memory DuckDB database");
        let reader = DuckDBReader::from_connection_string("duckdb://memory")?;

        // Load sample data if requested
        if cli.load_sample_data {
            info!("Loading sample data (products, sales, employees tables)");
            load_sample_data(&reader)?;
        }

        // Load data files if provided
        if !cli.load_data_files.is_empty() {
            info!("Loading {} data file(s)", cli.load_data_files.len());
            load_data_files(&reader, &cli.load_data_files)?;
        }

        Some(std::sync::Arc::new(std::sync::Mutex::new(reader)))
    } else {
        info!("Starting with empty in-memory database (no data pre-loaded)");
        None
    };

    #[cfg(not(feature = "duckdb"))]
    let reader = None;

    // Create application state
    let state = AppState {
        #[cfg(feature = "duckdb")]
        reader,
    };

    // Configure CORS
    let cors = if cli.cors_origin == "*" {
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(vec![header::CONTENT_TYPE])
    } else {
        let origins: Vec<_> = cli
            .cors_origin
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        CorsLayer::new()
            .allow_origin(origins)
            .allow_methods(Any)
            .allow_headers(vec![header::CONTENT_TYPE])
    };

    // Build router
    let app = Router::new()
        .route("/", get(root_handler))
        .route("/api/v1/query", post(query_handler))
        .route("/api/v1/parse", post(parse_handler))
        .route("/api/v1/health", get(health_handler))
        .route("/api/v1/version", get(version_handler))
        .layer(cors)
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .with_state(state);

    // Parse bind address
    let addr: SocketAddr = format!("{}:{}", cli.host, cli.port)
        .parse()
        .expect("Invalid host or port");

    info!("Starting ggsql REST API server on {}", addr);
    info!("API documentation:");
    info!("  POST /api/v1/query  - Execute ggsql query");
    info!("  POST /api/v1/parse  - Parse ggsql query");
    info!("  GET  /api/v1/health - Health check");
    info!("  GET  /api/v1/version - Version info");

    // Start server
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
