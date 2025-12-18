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
                eprintln!("Reader: {}", reader);
                eprintln!("Writer: {}", writer);
                if let Some(ref output_file) = output {
                    eprintln!("Output: {}", output_file.display());
                }
            }

            // Split query into SQL and ggSQL portions
            match parser::split_query(&query) {
                Ok((sql_part, viz_part)) => {
                    if verbose {
                        eprintln!("\nQuery split:");
                        eprintln!("  SQL portion: {} chars", sql_part.len());
                        eprintln!("  ggSQL portion: {} chars", viz_part.len());
                    }

                    // Execute SQL portion using the reader
                    #[cfg(feature = "duckdb")]
                    if reader.starts_with("duckdb://") {
                        match DuckDBReader::from_connection_string(&reader) {
                            Ok(db_reader) => {
                                match db_reader.execute(&sql_part) {
                                    Ok(df) => {
                                        if verbose {
                                            eprintln!("\nQuery executed successfully!");
                                            eprintln!("Result shape: {:?}", df.shape());
                                            eprintln!("Columns: {:?}", df.get_column_names());
                                        }

                                        // Parse ggSQL portion
                                        match parser::parse_query(&query) {
                                            Ok(specs) => {
                                                if verbose {
                                                    eprintln!("\nParsed {} visualization spec(s)", specs.len());
                                                }

                                                // Generate visualization output using writer
                                                #[cfg(feature = "vegalite")]
                                                if writer == "vegalite" {
                                                    let vl_writer = VegaLiteWriter::new();

                                                    // For now, render the first spec only
                                                    if let Some(spec) = specs.first() {
                                                        match vl_writer.write(spec, &df) {
                                                            Ok(json_output) => {
                                                                if let Some(output_path) = &output {
                                                                    // Write to file
                                                                    match std::fs::write(output_path, &json_output) {
                                                                        Ok(_) => {
                                                                            if verbose {
                                                                                eprintln!("\nVega-Lite JSON written to: {}", output_path.display());
                                                                            }
                                                                        }
                                                                        Err(e) => {
                                                                            eprintln!("Failed to write output file: {}", e);
                                                                            std::process::exit(1);
                                                                        }
                                                                    }
                                                                } else {
                                                                    // Print to stdout (always output the result)
                                                                    println!("{}", json_output);
                                                                }
                                                            }
                                                            Err(e) => {
                                                                eprintln!("Failed to generate Vega-Lite output: {}", e);
                                                                std::process::exit(1);
                                                            }
                                                        }
                                                    } else {
                                                        eprintln!("No visualization specifications found");
                                                        std::process::exit(1);
                                                    }
                                                }

                                                #[cfg(not(feature = "vegalite"))]
                                                {
                                                    if writer == "vegalite" {
                                                        eprintln!("VegaLite writer not compiled in. Rebuild with --features vegalite");
                                                        std::process::exit(1);
                                                    }
                                                }

                                                if writer != "vegalite" {
                                                    eprintln!("\nNote: Writer '{}' not yet implemented", writer);
                                                    eprintln!("Available writers: vegalite");
                                                }
                                            }
                                            Err(e) => {
                                                eprintln!("Failed to parse ggSQL portion: {}", e);
                                                std::process::exit(1);
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("Failed to execute SQL query: {}", e);
                                        std::process::exit(1);
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("Failed to create DuckDB reader: {}", e);
                                std::process::exit(1);
                            }
                        }
                    } else {
                        eprintln!("Unsupported reader: {}", reader);
                        eprintln!("Currently only 'duckdb://' readers are supported");
                        std::process::exit(1);
                    }

                    #[cfg(not(feature = "duckdb"))]
                    {
                        eprintln!("No reader support compiled in. Rebuild with --features duckdb");
                        std::process::exit(1);
                    }
                }
                Err(e) => {
                    eprintln!("Failed to split query: {}", e);
                    std::process::exit(1);
                }
            }
        }

        Commands::Run { file, reader, writer, output, verbose } => {
            if verbose {
                eprintln!("Running query from file: {}", file.display());
                eprintln!("Reader: {}", reader);
                eprintln!("Writer: {}", writer);
                if let Some(ref output_file) = output {
                    eprintln!("Output: {}", output_file.display());
                }
            }

            // Read query from file
            match std::fs::read_to_string(&file) {
                Ok(query) => {
                    // Execute the query (reuse exec logic)
                    match parser::split_query(&query) {
                        Ok((sql_part, viz_part)) => {
                            if verbose {
                                eprintln!("\nQuery split:");
                                eprintln!("  SQL portion: {} chars", sql_part.len());
                                eprintln!("  ggSQL portion: {} chars", viz_part.len());
                            }

                            // Execute SQL portion using the reader
                            #[cfg(feature = "duckdb")]
                            if reader.starts_with("duckdb://") {
                                match DuckDBReader::from_connection_string(&reader) {
                                    Ok(db_reader) => {
                                        match db_reader.execute(&sql_part) {
                                            Ok(df) => {
                                                if verbose {
                                                    eprintln!("\nQuery executed successfully!");
                                                    eprintln!("Result shape: {:?}", df.shape());
                                                    eprintln!("Columns: {:?}", df.get_column_names());
                                                }

                                                // Parse ggSQL portion
                                                match parser::parse_query(&query) {
                                                    Ok(specs) => {
                                                        if verbose {
                                                            eprintln!("\nParsed {} visualization spec(s)", specs.len());
                                                        }

                                                        // Generate visualization output using writer
                                                        #[cfg(feature = "vegalite")]
                                                        if writer == "vegalite" {
                                                            let vl_writer = VegaLiteWriter::new();

                                                            // For now, render the first spec only
                                                            if let Some(spec) = specs.first() {
                                                                match vl_writer.write(spec, &df) {
                                                                    Ok(json_output) => {
                                                                        if let Some(output_path) = &output {
                                                                            // Write to file
                                                                            match std::fs::write(output_path, &json_output) {
                                                                                Ok(_) => {
                                                                                    if verbose {
                                                                                        eprintln!("\nVega-Lite JSON written to: {}", output_path.display());
                                                                                    }
                                                                                }
                                                                                Err(e) => {
                                                                                    eprintln!("Failed to write output file: {}", e);
                                                                                    std::process::exit(1);
                                                                                }
                                                                            }
                                                                        } else {
                                                                            // Print to stdout (always output the result)
                                                                            println!("{}", json_output);
                                                                        }
                                                                    }
                                                                    Err(e) => {
                                                                        eprintln!("Failed to generate Vega-Lite output: {}", e);
                                                                        std::process::exit(1);
                                                                    }
                                                                }
                                                            } else {
                                                                eprintln!("No visualization specifications found");
                                                                std::process::exit(1);
                                                            }
                                                        }

                                                        #[cfg(not(feature = "vegalite"))]
                                                        {
                                                            if writer == "vegalite" {
                                                                eprintln!("VegaLite writer not compiled in. Rebuild with --features vegalite");
                                                                std::process::exit(1);
                                                            }
                                                        }

                                                        if writer != "vegalite" {
                                                            eprintln!("\nNote: Writer '{}' not yet implemented", writer);
                                                            eprintln!("Available writers: vegalite");
                                                        }
                                                    }
                                                    Err(e) => {
                                                        eprintln!("Failed to parse ggSQL portion: {}", e);
                                                        std::process::exit(1);
                                                    }
                                                }
                                            }
                                            Err(e) => {
                                                eprintln!("Failed to execute SQL query: {}", e);
                                                std::process::exit(1);
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("Failed to create DuckDB reader: {}", e);
                                        std::process::exit(1);
                                    }
                                }
                            } else {
                                eprintln!("Unsupported reader: {}", reader);
                                eprintln!("Currently only 'duckdb://' readers are supported");
                                std::process::exit(1);
                            }

                            #[cfg(not(feature = "duckdb"))]
                            {
                                eprintln!("No reader support compiled in. Rebuild with --features duckdb");
                                std::process::exit(1);
                            }
                        }
                        Err(e) => {
                            eprintln!("Failed to split query: {}", e);
                            std::process::exit(1);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Failed to read file {}: {}", file.display(), e);
                    std::process::exit(1);
                }
            }
        }

        Commands::Parse { query, format } => {
            println!("Parsing query: {}", query);
            println!("Format: {}", format);
            // TODO: Implement parsing logic
            match parser::parse_query(&query) {
                Ok(specs) => {
                    match format.as_str() {
                        "json" => println!("{}", serde_json::to_string_pretty(&specs)?),
                        "debug" => println!("{:#?}", specs),
                        "pretty" => {
                            println!("ggSQL Specifications: {} total", specs.len());
                            for (i, spec) in specs.iter().enumerate() {
                                println!("\nVisualization #{}:", i + 1);
                                println!("  Global Mapping: {:?}", spec.global_mapping);
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
                Err(e) => {
                    eprintln!("Parse error: {}", e);
                    std::process::exit(1);
                }
            }
        }

        Commands::Validate { query, reader } => {
            println!("Validating query: {}", query);
            if let Some(reader) = reader {
                println!("Reader: {}", reader);
            }
            // TODO: Implement validation logic
            println!("Validation not yet implemented");
        }
    }

    Ok(())
}