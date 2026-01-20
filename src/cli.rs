/*!
ggsql Command Line Interface

Provides commands for executing ggsql queries with various data sources and output formats.
*/

use clap::{Parser, Subcommand};
use ggsql::parser::extract_sql;
use ggsql::{parser, VERSION};
use std::path::PathBuf;

#[cfg(feature = "duckdb")]
use ggsql::execute::prepare_data;
#[cfg(feature = "duckdb")]
use ggsql::reader::{DuckDBReader, Reader};

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

    /// Execute a ggsql query from a file
    Run {
        /// Path to .sql file containing ggsql query
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

        /// Data source connection string (needed for column validation)
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

    // Check if visualise part is empty
    let parsed = parser::split_query(&query);
    if let Err(e) = parsed {
        eprintln!("Failed to split query: {}", e);
        std::process::exit(1);
    }
    let (_, viz_part) = parsed.unwrap();

    if viz_part.is_empty() {
        if verbose {
            eprintln!("Visualisation is empty. Printing table instead.");
        }
        print_table_fallback(&query, &db_reader, 100);
        return;
    }

    // Prepare data (parses query, executes SQL, handles layer sources)
    let prepared = prepare_data(&query, &db_reader);
    if let Err(e) = prepared {
        eprintln!("Failed to prepare data: {}", e);
        std::process::exit(1);
    }
    let prepared = prepared.unwrap();

    if verbose {
        eprintln!("\nData sources loaded:");
        for (key, df) in &prepared.data {
            eprintln!("  {}: {:?}", key, df.shape());
        }
        eprintln!("\nParsed {} visualisation spec(s)", prepared.specs.len());
    }

    let first_spec = prepared.specs.first();
    if first_spec.is_none() {
        eprintln!("No visualization specifications found");
        std::process::exit(1);
    }
    let first_spec = first_spec.unwrap();

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

    // Write visualization
    let vl_writer = VegaLiteWriter::new();
    let json_output = vl_writer.write(first_spec, &prepared.data);
    if let Err(ref e) = json_output {
        eprintln!("Failed to generate Vega-Lite output: {}", e);
        std::process::exit(1);
    }
    let json_output = json_output.unwrap();

    if output.is_none() {
        // Empty output location, write to stdout
        println!("{}", json_output);
        return;
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

// Prints a CSV-like output to stdout with aligned columns
fn print_table_fallback(query: &str, reader: &DuckDBReader, max_rows: usize) {
    let parsed = extract_sql(query);
    if let Err(e) = parsed {
        eprintln!("Failed to split query: {}", e);
        std::process::exit(1);
    }
    let parsed = parsed.unwrap();

    let data = reader.execute(&parsed);
    if let Err(e) = data {
        eprintln!("Failed to execute SQL query: {}", e);
        std::process::exit(1)
    }
    let data = data.unwrap();

    let nrow = data.height().min(max_rows);
    let ncol = data.width();
    let colnames = data.get_column_names_str();

    // We add an extra 'row' for the column names
    let mut rows: Vec<String> = vec![String::from(""); nrow + 1];

    for col_id in 0..ncol {
        let col_name = colnames[col_id];
        let mut width = col_name.chars().count();

        // End last column without comma
        let mut suffix = ", ";
        if col_id == ncol - 1 {
            suffix = "";
        }

        // Prepopulate formatted column with column name
        let mut col_fmt: Vec<String> = vec![format!("{}{}", col_name, suffix)];

        // Format every cell in column, tracking width
        let column_data = &data[col_id];
        for cell in column_data.iter().take(rows.len()) {
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
        for i in 0..rows.len() {
            rows[i].push_str(col_fmt[i].as_str());
        }
    }

    let output = rows.join("\n");
    println!("{}", output);
}
