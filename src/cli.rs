/*!
ggSQL Command Line Interface

Provides commands for executing ggSQL queries with various data sources and output formats.
*/

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use ggsql::{parser, VERSION};

#[cfg(feature = "duckdb")]
use ggsql::reader::{Reader, DuckDBReader};

#[cfg(feature = "vegalite")]
use ggsql::writer::{Writer, VegaLiteWriter};

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
    /// Execute a ggSQL query
    Exec {
        /// The ggSQL query to execute
        query: String,

        /// Data source connection string
        #[arg(long, default_value = "duckdb://memory")]
        reader: String,

        /// Output format
        #[arg(long, default_value = "vegalite")]
        writer: String,

        /// Output file path
        #[arg(long)]
        output: Option<PathBuf>,

        /// Show verbose output (execution details, statistics)
        #[arg(short, long)]
        verbose: bool,
    },

    /// Execute a ggSQL query from a file
    Run {
        /// Path to .sql file containing ggSQL query
        file: PathBuf,

        /// Data source connection string
        #[arg(long, default_value = "duckdb://memory")]
        reader: String,

        /// Output format
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
        /// The ggSQL query to parse
        query: String,

        /// Output format for AST (json, debug, pretty)
        #[arg(long, default_value = "pretty")]
        format: String,
    },

    /// Validate a query without executing
    Validate {
        /// The ggSQL query to validate
        query: String,

        /// Data source connection string (needed for column validation)
        #[arg(long)]
        reader: Option<String>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Exec { query, reader, writer, output, verbose } => {
            if verbose {
                eprintln!("Executing query: {}", query);
            }
            cmd_exec(query, reader, writer, output, verbose);
        }

        Commands::Run { file, reader, writer, output, verbose } => {
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
        Ok(query) => {
            cmd_exec(query, reader, writer, output, verbose)
        }
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

    // Split query into sql and ggsql part
    let parsed = parser::split_query(&query);
    if let Err(e) = parsed {
        eprintln!("Failed to split query: {}", e);
        std::process::exit(1);
    }
    let (sql_part, viz_part) = parsed.unwrap();

    if verbose {
        eprintln!("\nQuery split:");
        eprintln!("  SQL portion: {} chars", sql_part.len());
        eprintln!("  ggSQL portion: {} chars", viz_part.len());
    }

    // Setup reader
    #[cfg(feature = "duckdb")]
    if !reader.starts_with("duckdb://") {
        eprintln!("Unsupported reader: {}", reader);
        eprintln!("Currently only 'duckdb://' readers are supported");
        std::process::exit(1);
    }

    let db_reader = DuckDBReader::from_connection_string(&reader);
    if let Err(e) = db_reader {
        eprintln!("Failed to create DuckDB reader: {}", e);
        std::process::exit(1);
    }
    let db_reader = db_reader.unwrap();

    // Execute sql query
    let sql_result = db_reader.execute(&sql_part);
    if let Err(e) = sql_result {
        eprintln!("Failed to execute SQL query: {}", e);
        std::process::exit(1);
    }
    let df = sql_result.unwrap();

    if verbose {
        eprintln!("\nQuery executed successfully!");
        eprintln!("Result shape: {:?}", df.shape());
        eprintln!("Columns: {:?}", df.get_column_names());
    }

    // Parse ggSQL portion
    if viz_part.len() < 1 {
        if verbose {
            eprintln!("The ggSQL portion is empty. No specifications produced.");
        }
        return ();
    }

    let parsed = parser::parse_query(&query);
    if let Err(e) = parsed {
        eprintln!("Failed to parse ggSQL portion: {}", e);
        std::process::exit(1);
    }
    let specs = parsed.unwrap(); 

    let first_spec = specs.first();
    if let None = first_spec {
        eprintln!("No visualization specifications found");
        std::process::exit(1);
    }
    let first_spec = first_spec.unwrap();

    if verbose {
        eprintln!("\nParsed {} visualisation spec(s)", specs.len());
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

    // Write
    let vl_writer = VegaLiteWriter::new();
    let json_output = vl_writer.write(first_spec, &df);
    if let Err(ref e) = json_output {
        eprintln!("Failed to generate Vega-Lite output: {}", e);
    }
    let json_output = json_output.unwrap();

    if let None = output {
        // Empty output location, write to stdout
        println!("{}", json_output);
        return ();
    }
    let output = output.unwrap();

    // Write to file
    match std::fs::write(&output, &json_output) {
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
        "json" => {
            match serde_json::to_string_pretty(&specs) {
                Ok(pretty) => println!("{}", pretty),
                Err(error) => eprintln!("{}", error)
            }
        }
        "debug" => println!("{:#?}", specs),
        "pretty" => {
            println!("ggSQL Specifications: {} total", specs.len());
            for (i, spec) in specs.iter().enumerate() {
                println!("\nVisualization #{} ({:?}):", i + 1, spec.viz_type);
                println!("  Layers: {}", spec.layers.len());
                println!("  Scales: {}", spec.scales.len());
                if spec.facet.is_some() {
                    println!("  Faceting: Yes");
                }
                if spec.theme.is_some() {
                    println!("  Theme: Yes");
                }
            }
        }
        _ => {
            eprintln!("Unknown format: {}", format);
            std::process::exit(1);
        }
    }
}

fn cmd_validate(query: String, reader: Option<String>) {
    println!("Validating query: {}", query);
    if let Some(reader) = reader {
        println!("Reader: {}", reader);
    }
    // TODO: Implement validation logic
    println!("Validation not yet implemented");
}