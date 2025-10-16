/*!
VizQL Command Line Interface

Provides commands for executing VizQL queries with various data sources and output formats.
*/

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use vizql::{parser, VERSION};

#[derive(Parser)]
#[command(name = "vizql")]
#[command(about = "SQL extension for declarative data visualization")]
#[command(version = VERSION)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Execute a VizQL query
    Exec {
        /// The VizQL query to execute
        query: String,

        /// Data source connection string
        #[arg(long, default_value = "duckdb://memory")]
        reader: String,

        /// Output format
        #[arg(long, default_value = "ggplot2")]
        writer: String,

        /// Output file path
        #[arg(long)]
        output: Option<PathBuf>,
    },

    /// Execute a VizQL query from a file
    Run {
        /// Path to .sql file containing VizQL query
        file: PathBuf,

        /// Data source connection string
        #[arg(long, default_value = "duckdb://memory")]
        reader: String,

        /// Output format
        #[arg(long, default_value = "ggplot2")]
        writer: String,

        /// Output file path
        #[arg(long)]
        output: Option<PathBuf>,
    },

    /// Parse a query and show the AST (for debugging)
    Parse {
        /// The VizQL query to parse
        query: String,

        /// Output format for AST (json, debug, pretty)
        #[arg(long, default_value = "pretty")]
        format: String,
    },

    /// Validate a query without executing
    Validate {
        /// The VizQL query to validate
        query: String,

        /// Data source connection string (needed for column validation)
        #[arg(long)]
        reader: Option<String>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Exec { query, reader, writer, output } => {
            println!("Executing query: {}", query);
            println!("Reader: {}", reader);
            println!("Writer: {}", writer);
            if let Some(output) = output {
                println!("Output: {}", output.display());
            }
            // TODO: Implement execution logic
            println!("Execution not yet implemented");
        }

        Commands::Run { file, reader, writer, output } => {
            println!("Running query from file: {}", file.display());
            println!("Reader: {}", reader);
            println!("Writer: {}", writer);
            if let Some(output) = output {
                println!("Output: {}", output.display());
            }
            // TODO: Implement file execution logic
            println!("File execution not yet implemented");
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
                            println!("VizQL Specifications: {} total", specs.len());
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