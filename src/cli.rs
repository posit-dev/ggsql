/*!
ggsql Command Line Interface

Provides commands for executing ggsql queries with various data sources and output formats.
*/

use clap::{Parser, Subcommand};
use ggsql::reader::{Reader, Spec};
use ggsql::validate::validate;
use ggsql::{parser, VERSION};
use std::path::PathBuf;

#[cfg(feature = "vegalite")]
use ggsql::writer::{VegaLiteWriter, Writer};

#[derive(Parser)]
#[command(name = "ggsql")]
#[command(about = "SQL extension for declarative data visualization")]
#[command(version = VERSION)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Execute a ggsql query
    Exec {
        /// The ggsql query to execute
        query: String,

        /// Data source connection string (duckdb://, sqlite://, odbc://)
        #[arg(long, default_value = "duckdb://memory")]
        reader: String,

        /// Output format (vegalite)
        #[arg(long, default_value = "vegalite")]
        writer: String,

        /// Output file path
        #[arg(long)]
        output: Option<PathBuf>,

        /// Show verbose output (execution details, statistics)
        #[arg(short, long)]
        verbose: bool,
    },

    /// Execute a ggsql query from a file
    Run {
        /// Path to .sql file containing ggsql query
        file: PathBuf,

        /// Data source connection string (duckdb://, sqlite://, odbc://)
        #[arg(long, default_value = "duckdb://memory")]
        reader: String,

        /// Output format (vegalite)
        #[arg(long, default_value = "vegalite")]
        writer: String,

        /// Output file path
        #[arg(long)]
        output: Option<PathBuf>,

        /// Show verbose output (execution details, statistics)
        #[arg(short, long)]
        verbose: bool,
    },

    /// Parse a query and show the AST (for debugging)
    Parse {
        /// The ggsql query to parse
        query: String,

        /// Output format for AST (json, debug, pretty)
        #[arg(long, default_value = "pretty")]
        format: String,
    },

    /// Validate a query without executing
    Validate {
        /// The ggsql query to validate
        query: String,

        /// Data source connection string for column validation (duckdb://, sqlite://, polars://)
        #[arg(long)]
        reader: Option<String>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Exec {
            query,
            reader,
            writer,
            output,
            verbose,
        } => {
            if verbose {
                eprintln!("Executing query: {}", query);
            }
            cmd_exec(query, reader, writer, output, verbose);
        }

        Commands::Run {
            file,
            reader,
            writer,
            output,
            verbose,
        } => {
            if verbose {
                eprintln!("Running query from file: {}", file.display());
            }
            cmd_run(file, reader, writer, output, verbose);
        }

        Commands::Parse { query, format } => {
            cmd_parse(query, format);
        }

        Commands::Validate { query, reader } => {
            cmd_validate(query, reader);
        }
    }

    Ok(())
}

fn cmd_run(file: PathBuf, reader: String, writer: String, output: Option<PathBuf>, verbose: bool) {
    match std::fs::read_to_string(&file) {
        Ok(query) => cmd_exec(query, reader, writer, output, verbose),
        Err(e) => {
            eprintln!("Failed to read file {}: {}", file.display(), e);
            std::process::exit(1);
        }
    }
}

fn cmd_exec(query: String, reader: String, writer: String, output: Option<PathBuf>, verbose: bool) {
    if verbose {
        eprintln!("Reader: {}", reader);
        eprintln!("Writer: {}", writer);
        if let Some(ref output_file) = output {
            eprintln!("Output: {}", output_file.display());
        }
    }

    if reader.starts_with("duckdb://") {
        #[cfg(feature = "duckdb")]
        {
            let r = match ggsql::reader::DuckDBReader::from_connection_string(&reader) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Failed to create reader: {}", e);
                    std::process::exit(1);
                }
            };
            exec_with_reader(&query, &r, &writer, output, verbose);
        }
        #[cfg(not(feature = "duckdb"))]
        {
            eprintln!("DuckDB reader not compiled in. Rebuild with --features duckdb");
            std::process::exit(1);
        }
    } else if reader.starts_with("sqlite://") {
        #[cfg(feature = "sqlite")]
        {
            let r = match ggsql::reader::SqliteReader::from_connection_string(&reader) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Failed to create reader: {}", e);
                    std::process::exit(1);
                }
            };
            exec_with_reader(&query, &r, &writer, output, verbose);
        }
        #[cfg(not(feature = "sqlite"))]
        {
            eprintln!("SQLite reader not compiled in. Rebuild with --features sqlite");
            std::process::exit(1);
        }
    } else if reader.starts_with("odbc://") {
        #[cfg(feature = "odbc")]
        {
            let r = match ggsql::reader::OdbcReader::from_connection_string(&reader) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Failed to create reader: {}", e);
                    std::process::exit(1);
                }
            };
            exec_with_reader(&query, &r, &writer, output, verbose);
        }
        #[cfg(not(feature = "odbc"))]
        {
            eprintln!("ODBC reader not compiled in. Rebuild with --features odbc");
            std::process::exit(1);
        }
    } else if reader.starts_with("postgres://") || reader.starts_with("postgresql://") {
        eprintln!("PostgreSQL reader is not yet implemented");
        std::process::exit(1);
    } else {
        eprintln!("Unsupported connection string: {}", reader);
        std::process::exit(1);
    }
}

fn exec_with_reader<R: Reader>(
    query: &str,
    reader: &R,
    writer: &str,
    output: Option<PathBuf>,
    verbose: bool,
) {
    // Use validate() to check if query has visualization
    let validated = match validate(query) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to validate query: {}", e);
            std::process::exit(1);
        }
    };

    if !validated.has_visual() {
        if verbose {
            eprintln!("Visualisation is empty. Printing table instead.");
        }
        print_table_fallback(query, reader, 100);
        return;
    }

    // Execute ggsql query
    let spec = match reader.execute(query) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to execute query: {}", e);
            std::process::exit(1);
        }
    };

    render_spec(spec, writer, output, verbose);
}

fn render_spec(spec: Spec, writer: &str, output: Option<PathBuf>, verbose: bool) {
    if verbose {
        let metadata = spec.metadata();
        eprintln!("\nQuery executed:");
        eprintln!("  Rows: {}", metadata.rows);
        eprintln!("  Columns: {}", metadata.columns.join(", "));
        eprintln!("  Layers: {}", metadata.layer_count);
    }

    if spec.plot().layers.is_empty() {
        eprintln!("No visualization specifications found");
        std::process::exit(1);
    }

    // Check writer
    if writer != "vegalite" {
        eprintln!("\nNote: Writer '{}' not yet implemented", writer);
        eprintln!("Available writers: vegalite")
    }

    #[cfg(not(feature = "vegalite"))]
    {
        eprintln!("VegaLite writer not compiled in. Rebuild with --features vegalite");
        std::process::exit(1)
    }

    // Render
    let vl_writer = VegaLiteWriter::new();
    let json_output = match vl_writer.render(&spec) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to generate Vega-Lite output: {}", e);
            std::process::exit(1);
        }
    };

    if output.is_none() {
        // Empty output location, write to stdout
        println!("{}", json_output);
        return;
    }
    let output = output.unwrap();

    // Write to file
    match std::fs::write(&output, json_output) {
        Ok(_) => {
            if verbose {
                eprintln!("\nVega-Lite JSON written to: {}", output.display());
            }
        }
        Err(e) => {
            eprintln!("Failed to write to output file: {}", e);
            std::process::exit(1);
        }
    }
}

fn cmd_parse(query: String, format: String) {
    println!("Parsing query: {}", query);
    println!("Format: {}", format);

    let parsed = parser::parse_query(&query);

    if let Err(e) = parsed {
        eprintln!("Parse error: {}", e);
        std::process::exit(1);
    }
    // TODO: implement parsing logic
    let specs = parsed.unwrap();

    match format.as_str() {
        "json" => match serde_json::to_string_pretty(&specs) {
            Ok(pretty) => println!("{}", pretty),
            Err(error) => eprintln!("{}", error),
        },
        "debug" => println!("{:#?}", specs),
        "pretty" => {
            println!("ggsql Specifications: {} total", specs.len());
            for (i, spec) in specs.iter().enumerate() {
                println!("\nVisualization #{}:", i + 1);
                println!("  Global Mappings: {:?}", spec.global_mappings);
                println!("  Layers: {}", spec.layers.len());
                println!("  Scales: {}", spec.scales.len());
                if spec.facet.is_some() {
                    println!("  Faceting: Yes");
                }
            }
        }
        _ => {
            eprintln!("Unknown format: {}", format);
            std::process::exit(1);
        }
    }
}

fn cmd_validate(query: String, _reader: Option<String>) {
    match validate(&query) {
        Ok(validated) if validated.valid() => {
            println!("✓ Query syntax is valid");
        }
        Ok(validated) => {
            println!("✗ Validation errors:");
            for err in validated.errors() {
                println!("  - {}", err.message);
            }
            if !validated.warnings().is_empty() {
                println!("\nWarnings:");
                for warning in validated.warnings() {
                    println!("  - {}", warning.message);
                }
            }
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("Error during validation: {}", e);
            std::process::exit(1);
        }
    }
}

// Prints a CSV-like output to stdout with aligned columns
fn print_table_fallback<R: Reader>(query: &str, reader: &R, max_rows: usize) {
    let source_tree = match parser::SourceTree::new(query) {
        Ok(st) => st,
        Err(e) => {
            eprintln!("Failed to parse query: {}", e);
            std::process::exit(1);
        }
    };

    let sql_part = source_tree.extract_sql().unwrap_or_default();

    let data = reader.execute_sql(&sql_part);
    if let Err(e) = data {
        eprintln!("Failed to execute SQL query: {}", e);
        std::process::exit(1)
    }
    let data = data.unwrap();

    let nrow = data.height().min(max_rows);
    let ncol = data.width();
    let colnames = data.get_column_names();

    // We add an extra 'row' for the column names
    let mut rows: Vec<String> = vec![String::from(""); nrow + 1];

    let columns = data.get_columns();
    for (col_id, (col_name, column_data)) in
        colnames.iter().zip(columns.iter()).enumerate()
    {
        let mut width = col_name.chars().count();

        // End last column without comma
        let suffix = if col_id == ncol - 1 { "" } else { ", " };

        // Prepopulate formatted column with column name
        let mut col_fmt: Vec<String> = vec![format!("{}{}", col_name, suffix)];

        // Format every cell in column, tracking width
        for row_idx in 0..nrow {
            let cell = ggsql::array_util::value_to_string(column_data, row_idx);
            let cell_fmt = format!("{}{}", cell, suffix);
            let nchar = cell_fmt.chars().count();
            if nchar > width {
                width = nchar;
            }
            col_fmt.push(cell_fmt);
        }
        // Pad strings with spaces
        let col_fmt: Vec<String> = col_fmt
            .into_iter()
            .map(|s| format!("{:width$}", s, width = width))
            .collect();

        // Push columns to row string
        for (row, fmt) in rows.iter_mut().zip(col_fmt.iter()) {
            row.push_str(fmt.as_str());
        }
    }

    let output = rows.join("\n");
    println!("{}", output);
}
