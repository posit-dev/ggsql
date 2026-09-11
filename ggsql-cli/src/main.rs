/*!
ggsql Command Line Interface

Provides commands for executing ggsql queries with various data sources and output formats.
*/

use clap::{Args, Parser, Subcommand, ValueEnum};
use ggsql::reader::{connection, Reader, ResolvedSpec};
use ggsql::validate::validate;
use ggsql::writer::WriterOptions;
use ggsql::{parser, VERSION};
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use writers::{Output, WriterInfo};

mod writers;

mod docs {
    include!(concat!(env!("OUT_DIR"), "/docs_data.rs"));
}

#[derive(Parser)]
#[command(name = "ggsql")]
#[command(about = "SQL extension for declarative data visualization")]
#[command(version = VERSION)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// The writer to render with, plus the `--writer-option` settings for it.
struct WriterSpec {
    info: &'static WriterInfo,
    options: WriterOptions,
}

/// Where a subcommand's data comes from.
///
/// Flattened into both [`RenderArgs`] and [`ViewArgs`], so the two flags are
/// declared — and helped, and defaulted — once: where a plot's data comes from
/// does not depend on whether the plot ends up in a file or in a window.
#[derive(Args)]
pub struct ReaderArgs {
    /// Data source connection string (duckdb://, sqlite://, odbc://)
    #[arg(short, long, default_value = "duckdb://memory")]
    pub reader: String,

    /// In-memory cache backend wrapping the reader (duckdb, sqlite). Off by default.
    #[arg(long)]
    pub cache: Option<String>,
}

/// The flags shared by `exec` and `run`: where the data comes from, which
/// writer renders it, and where the result goes.
#[derive(Args)]
pub struct RenderArgs {
    #[command(flatten)]
    pub source: ReaderArgs,

    /// Output format — run with --help for the writers this build has
    ///
    /// Left unset, `--output`'s extension picks the writer, falling back to
    /// vegalite. `Option` rather than a clap `default_value` so "unset" stays
    /// distinguishable from "explicitly vegalite"; the long help states the
    /// default instead.
    #[arg(short, long, long_help = writers::writer_help())]
    pub writer: Option<String>,

    /// Settings for the chosen writer, as `key=value` (repeatable)
    #[arg(
        short = 'D',
        long = "writer-option",
        visible_alias = "writer-options",
        value_name = "KEY=VALUE[;...]",
        long_help = writers::option_help()
    )]
    pub writer_options: Vec<String>,

    /// Output file path
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Show verbose output (execution details, statistics)
    #[arg(short, long)]
    pub verbose: bool,
}

/// The flags `view` takes: where the data comes from and how the window looks.
///
/// Deliberately not [`RenderArgs`]: there is no `--writer` to choose and no
/// `--output` to write, and `-D` carries the viewer's own settings rather than
/// a writer's. The reader flags are the same ones, so they come from the same
/// [`ReaderArgs`].
#[derive(Args)]
pub struct ViewArgs {
    #[command(flatten)]
    pub source: ReaderArgs,

    /// Viewer settings, as `key=value` (repeatable)
    #[arg(
        short = 'D',
        long = "viewer-option",
        visible_alias = "viewer-options",
        value_name = "KEY=VALUE[;...]",
        long_help = "Settings for the viewer window, as `key=value`. Repeatable, and one flag \
                     may carry several settings separated by `;` (quote it, as most shells read \
                     `;` themselves): `-D 'width=1280;title=My plot'`.\n\nSettings:\n  \
                     width, height, background, title"
    )]
    pub viewer_options: Vec<String>,

    /// Show verbose output (execution details, statistics)
    #[arg(short, long)]
    pub verbose: bool,
}

impl RenderArgs {
    /// Resolve `--writer` and its settings, exiting on an unknown name or a
    /// malformed setting rather than discovering either after the SQL has run.
    fn writer(&self) -> WriterSpec {
        let info = self.resolve_writer();
        if !info.compiled {
            eprintln!("{}", writers::not_compiled_message(info));
            std::process::exit(1);
        }
        let options = WriterOptions::parse(self.writer_options.clone()).unwrap_or_else(|e| {
            eprintln!("{}", e);
            std::process::exit(1);
        });
        if let Err(e) = (info.check)(&options) {
            eprintln!("{}", e);
            std::process::exit(1);
        }
        WriterSpec { info, options }
    }

    /// Which writer to use: `--writer` if given, else what `--output`'s
    /// extension implies, else the default.
    ///
    /// An explicit `--writer` always wins; disagreeing with the extension only
    /// warns on stderr, since writing SVG to a `.txt` is legitimate.
    fn resolve_writer(&self) -> &'static writers::WriterInfo {
        if let Some(name) = &self.writer {
            let info = writers::find(name).unwrap_or_else(|| {
                eprintln!("{}", writers::unknown_writer(name));
                std::process::exit(1);
            });
            if let Some(implied) = self.output.as_deref().and_then(writers::for_extension) {
                if !std::ptr::eq(implied, info) {
                    eprintln!(
                        "Warning: writing {} to '{}', whose extension says {}",
                        info.label,
                        self.output.as_deref().unwrap_or(Path::new("")).display(),
                        implied.name
                    );
                }
            }
            return info;
        }
        // No --writer, so the extension decides. A writer it names but this
        // build lacks is an error, not a silent fallback to Vega-Lite JSON in
        // a file called `.png`.
        self.output
            .as_deref()
            .and_then(writers::for_extension)
            .unwrap_or_else(|| {
                writers::find(writers::DEFAULT_WRITER)
                    .expect("the default writer has a registry row")
            })
    }
}

#[derive(Subcommand)]
pub enum Commands {
    /// Execute a ggsql query
    Exec {
        /// The ggsql query to execute
        query: String,

        #[command(flatten)]
        render: RenderArgs,
    },

    /// Execute a ggsql query from a file
    Run {
        /// Path to .sql file containing ggsql query
        file: PathBuf,

        #[command(flatten)]
        render: RenderArgs,
    },

    /// Show a ggsql query's plot in a window
    ///
    /// Blocks until the window is closed. Resizing the window re-lays-out the
    /// plot rather than stretching it.
    ///
    /// Requires the `window` feature and a working GPU adapter.
    View {
        /// The ggsql query to show
        query: String,

        #[command(flatten)]
        view: ViewArgs,
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
        #[arg(short, long)]
        reader: Option<String>,
    },

    /// Show documentation for ggsql syntax (clauses, layers, scales, aesthetics, coords)
    ///
    /// Run `ggsql docs` with no arguments for an index of available topics.
    /// Clauses are looked up by name directly (e.g. `ggsql docs draw`).
    /// Other topics take a category first (e.g. `ggsql docs layer point`,
    /// `ggsql docs position stack`, `ggsql docs scale continuous`,
    /// `ggsql docs aesthetic color`, `ggsql docs coord cartesian`).
    Docs {
        /// Clause name (e.g. "draw") or category (e.g. "layer", "scale")
        first: Option<String>,

        /// Topic within the category (e.g. "point" when first is "layer")
        second: Option<String>,

        /// Output format. Defaults to rendered text on a TTY, raw markdown when piped.
        #[arg(long, value_enum)]
        format: Option<DocsFormat>,
    },

    /// Show the ggsql skill — a usage guide intended for AI assistants and humans
    ///
    /// The content is synced from https://github.com/posit-dev/skills at build time.
    Skill {
        /// Output format. Defaults to rendered text on a TTY, raw markdown when piped.
        #[arg(long, value_enum)]
        format: Option<DocsFormat>,
    },

    /// Alias for `skill` — show the ggsql usage guide for AI assistants
    #[command(name = "agent-info")]
    AgentInfo {
        /// Output format. Defaults to rendered text on a TTY, raw markdown when piped.
        #[arg(long, value_enum)]
        format: Option<DocsFormat>,
    },
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum DocsFormat {
    /// Markdown rendered to ANSI for terminal display
    Text,
    /// Raw markdown (ideal for piping or for agents)
    Markdown,
    /// Structured JSON with metadata and body
    Json,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Exec { query, render } => {
            if render.verbose {
                eprintln!("Executing query: {}", query);
            }
            let writer = render.writer();
            cmd_exec(query, &render, &writer);
        }

        Commands::Run { file, render } => {
            if render.verbose {
                eprintln!("Running query from file: {}", file.display());
            }
            let writer = render.writer();
            cmd_run(file, &render, &writer);
        }

        Commands::View { query, view } => {
            if view.verbose {
                eprintln!("Showing query: {}", query);
            }
            cmd_view(query, &view);
        }

        Commands::Parse { query, format } => {
            cmd_parse(query, format);
        }

        Commands::Validate { query, reader } => {
            cmd_validate(query, reader);
        }

        Commands::Docs {
            first,
            second,
            format,
        } => {
            cmd_docs(first, second, format);
        }

        Commands::Skill { format } | Commands::AgentInfo { format } => {
            cmd_skill(format);
        }
    }

    Ok(())
}

fn cmd_run(file: PathBuf, args: &RenderArgs, writer: &WriterSpec) {
    match std::fs::read_to_string(&file) {
        Ok(query) => cmd_exec(query, args, writer),
        Err(e) => {
            eprintln!("Failed to read file {}: {}", file.display(), e);
            std::process::exit(1);
        }
    }
}

fn cmd_exec(query: String, args: &RenderArgs, writer: &WriterSpec) {
    if args.verbose {
        eprintln!("Reader: {}", args.source.reader);
        if let Some(ref cache) = args.source.cache {
            eprintln!("Cache: {}", cache);
        }
        eprintln!("Writer: {}", writer.info.name);
        if let Some(ref output_file) = args.output {
            eprintln!("Output: {}", output_file.display());
        }
    }

    let reader =
        open_reader(&args.source.reader, args.source.cache.as_deref()).unwrap_or_else(|e| {
            eprintln!("{}", e);
            std::process::exit(1);
        });

    exec_with_reader(&query, reader.as_ref(), args, writer);
}

/// Open the reader named by a connection string, wrapped in a cache if asked
/// for one.
///
/// Which schemes exist, which of them this build has, and how a cache wraps a
/// primary all live in the library, as `connection::reader_from_uri`. What is
/// left here is the CLI's own spelling of it: `ggsql::reader::Reader` is
/// object-safe on purpose, so `exec`, `run` and `view` share one function.
///
/// `--cache <scheme>` is sugar for the composite `<scheme>+<primary>://` URI
/// `reader_from_uri` already understands. The two forms may not be combined —
/// there would be no saying which cache was meant.
fn open_reader(uri: &str, cache: Option<&str>) -> Result<Box<dyn Reader + Send>, String> {
    let uri = match cache {
        Some(cache_scheme) => {
            if connection::split_cache_uri(uri).is_some() {
                return Err("Cannot combine --cache with a composite \'<cache>+<primary>://\' connection string".to_string());
            }
            let Some((scheme, rest)) = uri.split_once("://") else {
                return Err(format!("Invalid --reader connection string: {uri}"));
            };
            format!("{cache_scheme}+{scheme}://{rest}")
        }
        None => uri.to_string(),
    };

    connection::reader_from_uri(&uri).map_err(|e| format!("Failed to create reader: {e}"))
}

fn exec_with_reader(query: &str, reader: &dyn Reader, args: &RenderArgs, writer: &WriterSpec) {
    // Use validate() to check if query has visualization
    let validated = match validate(query) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to validate query: {}", e);
            std::process::exit(1);
        }
    };

    if !validated.has_spec() {
        if args.verbose {
            eprintln!("Visualisation is empty. Printing table instead.");
        }
        print_table_fallback(query, reader, 100, args.output.as_deref());
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

    render_spec(spec, args, writer);
}

fn render_spec(spec: ResolvedSpec, args: &RenderArgs, writer: &WriterSpec) {
    match &spec {
        ResolvedSpec::Plot(plot) => {
            if args.verbose {
                let metadata = plot.metadata();
                eprintln!("\nQuery executed:");
                eprintln!("  Rows: {}", metadata.rows);
                eprintln!("  Columns: {}", metadata.columns.join(", "));
                eprintln!("  Layers: {}", metadata.layer_count);
            }

            if plot.plot().layers.is_empty() {
                eprintln!("No visualization specifications found");
                std::process::exit(1);
            }
        }
        ResolvedSpec::Table(table) => {
            if args.verbose {
                eprintln!("\nQuery executed:");
                eprintln!("  Rows: {}", table.body().height());
                eprintln!("  Columns: {}", table.body().width());
            }
        }
    }

    let info = writer.info;
    let (render, warnings) = (info.render)(&spec, &writer.options).unwrap_or_else(|e| {
        eprintln!("Failed to generate {} output: {}", info.label, e);
        std::process::exit(1);
    });

    // Not behind -v: a degraded render is a defect in the file about to be
    // shipped. stderr keeps it out of a piped artifact.
    for warning in &warnings {
        eprintln!("warning: {}", warning);
    }

    match (render, &args.output) {
        (Output::Text(txt), None) => {
            println!("{}", txt);
        }
        (Output::Text(txt), Some(path)) => match std::fs::write(path, txt) {
            Ok(_) => {
                if args.verbose {
                    eprintln!("\n{} written to: {}", info.label, path.display());
                }
            }
            Err(e) => {
                eprintln!("Failed to write to output file: {}", e);
                std::process::exit(1);
            }
        },
        (Output::Bin(buf), None) => {
            if std::io::stdout().is_terminal() {
                // Non-zero, since nothing was produced: `… && publish` must
                // not carry on as though it had a file.
                eprintln!("Suppressing output in terminal. Pipe output to another process or use --output <FILE> to save to a file.");
                std::process::exit(1);
            } else {
                std::io::stdout().write_all(&buf).unwrap_or_else(|e| {
                    eprintln!("Failed to write buffer with the error: {}", e);
                    std::process::exit(1);
                });
            }
        }
        (Output::Bin(buf), Some(path)) => match std::fs::write(path, buf) {
            Ok(_) => {
                if args.verbose {
                    eprintln!("\n{} written to: {}", info.label, path.display());
                }
            }
            Err(e) => {
                eprintln!("Failed to write to output file: {}", e);
                std::process::exit(1);
            }
        },
    };
}

/// Show a query's plot in a window, blocking until it closes.
///
/// The subcommand exists whether or not the feature does, so a build without
/// it says what would bring it back rather than dropping the command.
fn cmd_view(query: String, args: &ViewArgs) {
    #[cfg(feature = "window")]
    {
        use ggsql::writer::PlotViewer;

        let options = WriterOptions::parse(args.viewer_options.clone()).unwrap_or_else(|e| {
            eprintln!("{}", e);
            std::process::exit(1);
        });
        let viewer = PlotViewer::from_options(&options).unwrap_or_else(|e| {
            eprintln!("{}", e);
            std::process::exit(1);
        });

        if args.verbose {
            eprintln!("Reader: {}", args.source.reader);
            if let Some(ref cache) = args.source.cache {
                eprintln!("Cache: {}", cache);
            }
        }

        let reader =
            open_reader(&args.source.reader, args.source.cache.as_deref()).unwrap_or_else(|e| {
                eprintln!("{}", e);
                std::process::exit(1);
            });

        let validated = validate(&query).unwrap_or_else(|e| {
            eprintln!("Failed to validate query: {}", e);
            std::process::exit(1);
        });
        if !validated.has_spec() {
            eprintln!("This query has no VISUALISE clause, so there is no plot to show.");
            std::process::exit(1);
        }

        let spec = reader.execute(&query).unwrap_or_else(|e| {
            eprintln!("Failed to execute query: {}", e);
            std::process::exit(1);
        });
        let plot = spec.as_plot().unwrap_or_else(|| {
            eprintln!("This is a TABULATE query; there is no plot to show.");
            std::process::exit(1);
        });

        if args.verbose {
            let metadata = plot.metadata();
            eprintln!("  Rows: {}", metadata.rows);
            eprintln!("  Layers: {}", metadata.layer_count);
            eprintln!("Close the window to exit.");
        }

        // Blocks on the main thread until the window closes.
        if let Err(e) = viewer.show(plot) {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    }
    #[cfg(not(feature = "window"))]
    {
        let _ = (query, args);
        eprintln!("The plot viewer is not compiled in. Rebuild with --features window");
        std::process::exit(1);
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
                match spec {
                    ggsql::Spec::Plot(plot) => {
                        println!("\nVisualization #{}:", i + 1);
                        println!("  Global Mappings: {:?}", plot.global_mappings);
                        println!("  Layers: {}", plot.layers.len());
                        println!("  Scales: {}", plot.scales.len());
                        if plot.facet.is_some() {
                            println!("  Faceting: Yes");
                        }
                    }
                    ggsql::Spec::Table(_) => {
                        println!("\nTable #{}:", i + 1);
                    }
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

/// Print a query's table, for a query with nothing to draw.
///
/// CSV-like, with the columns aligned.
///
/// Honours `--output` like every other result — printing to stdout while
/// leaving the named file absent is hard to notice from a script.
fn print_table_fallback(
    query: &str,
    reader: &dyn Reader,
    max_rows: usize,
    output: Option<&std::path::Path>,
) {
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
    for (col_id, (col_name, column_data)) in colnames.iter().zip(columns.iter()).enumerate() {
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

    let table = rows.join("\n");
    match output {
        None => println!("{}", table),
        Some(path) => {
            if let Err(e) = std::fs::write(path, format!("{table}\n")) {
                eprintln!("Failed to write to output file: {}", e);
                std::process::exit(1);
            }
        }
    }
}

fn cmd_docs(first: Option<String>, second: Option<String>, format: Option<DocsFormat>) {
    let fmt = format.unwrap_or_else(|| {
        if std::io::stdout().is_terminal() {
            DocsFormat::Text
        } else {
            DocsFormat::Markdown
        }
    });

    match (first.as_deref(), second.as_deref()) {
        (None, _) => print_docs_index(fmt),
        (Some(arg), None) => {
            let arg_lc = arg.to_lowercase();
            if let Some(entry) = find_doc(None, &arg_lc) {
                render_doc(entry, fmt);
                return;
            }
            if is_category(&arg_lc) {
                print_category_listing(&arg_lc, fmt);
                return;
            }
            eprintln!("Unknown topic: {}", arg);
            eprintln!();
            print_docs_index_to(&mut std::io::stderr(), DocsFormat::Markdown);
            std::process::exit(1);
        }
        (Some(cat), Some(topic)) => {
            let cat_lc = cat.to_lowercase();
            let topic_lc = topic.to_lowercase();
            if let Some(entry) = find_doc(Some(&cat_lc), &topic_lc) {
                render_doc(entry, fmt);
            } else {
                eprintln!("Unknown topic: {} {}", cat, topic);
                eprintln!();
                print_category_listing_to(&mut std::io::stderr(), &cat_lc, DocsFormat::Markdown);
                std::process::exit(1);
            }
        }
    }
}

const CATEGORY_ORDER: &[(&str, &str)] = &[
    ("layer", "Layer types"),
    ("position", "Position adjustments"),
    ("scale", "Scale types"),
    ("aesthetic", "Aesthetics"),
    ("coord", "Coordinate systems"),
];

fn is_category(name: &str) -> bool {
    CATEGORY_ORDER.iter().any(|(cat, _)| *cat == name)
}

fn find_doc(category: Option<&str>, topic: &str) -> Option<&'static docs::DocEntry> {
    docs::DOCS
        .iter()
        .find(|e| e.category == category && e.topic.eq_ignore_ascii_case(topic))
}

fn topics_in(category: Option<&str>) -> Vec<&'static str> {
    docs::DOCS
        .iter()
        .filter(|e| e.category == category)
        .map(|e| e.topic)
        .collect()
}

fn strip_images(markdown: &str) -> String {
    use std::sync::OnceLock;
    static IMG_RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = IMG_RE.get_or_init(|| regex::Regex::new(r"!\[[^\]]*\]\(([^)]*)\)").unwrap());
    re.replace_all(markdown, "$1").to_string()
}

fn render_doc(entry: &docs::DocEntry, fmt: DocsFormat) {
    match fmt {
        DocsFormat::Text => {
            let skin = termimad::MadSkin::default();
            skin.print_text(&strip_images(entry.body));
        }
        DocsFormat::Markdown => {
            print!("{}", entry.body);
            if !entry.body.ends_with('\n') {
                println!();
            }
        }
        DocsFormat::Json => {
            let obj = serde_json::json!({
                "category": entry.category,
                "topic": entry.topic,
                "title": entry.title,
                "body": entry.body,
            });
            match serde_json::to_string_pretty(&obj) {
                Ok(s) => println!("{}", s),
                Err(e) => {
                    eprintln!("Failed to serialize docs entry: {}", e);
                    std::process::exit(1);
                }
            }
        }
    }
}

fn print_docs_index(fmt: DocsFormat) {
    let mut stdout = std::io::stdout();
    print_docs_index_to(&mut stdout, fmt);
}

fn print_docs_index_to<W: std::io::Write>(out: &mut W, fmt: DocsFormat) {
    if fmt == DocsFormat::Json {
        let mut sections = serde_json::Map::new();
        let clauses = topics_in(None);
        sections.insert("clauses".to_string(), serde_json::json!(clauses));
        for (cat, _) in CATEGORY_ORDER {
            sections.insert((*cat).to_string(), serde_json::json!(topics_in(Some(cat))));
        }
        let _ = writeln!(
            out,
            "{}",
            serde_json::to_string_pretty(&serde_json::Value::Object(sections)).unwrap()
        );
        return;
    }

    let clauses = topics_in(None);
    let _ = writeln!(out, "ggsql syntax reference");
    let _ = writeln!(out);
    let _ = writeln!(out, "Clauses         ggsql docs <name>");
    let _ = writeln!(out, "                {}", clauses.join(", "));
    let _ = writeln!(out);
    for (cat, label) in CATEGORY_ORDER {
        let topics = topics_in(Some(cat));
        if topics.is_empty() {
            continue;
        }
        let _ = writeln!(out, "{:<15} ggsql docs {} <name>", label, cat);
        let _ = writeln!(out, "                {}", topics.join(", "));
        let _ = writeln!(out);
    }
    let _ = writeln!(
        out,
        "Use `--format markdown` for raw markdown or `--format json` for structured output."
    );
}

fn print_category_listing(category: &str, fmt: DocsFormat) {
    let mut stdout = std::io::stdout();
    print_category_listing_to(&mut stdout, category, fmt);
}

fn print_category_listing_to<W: std::io::Write>(out: &mut W, category: &str, fmt: DocsFormat) {
    let topics = topics_in(Some(category));
    if fmt == DocsFormat::Json {
        let _ = writeln!(
            out,
            "{}",
            serde_json::json!({ "category": category, "topics": topics })
        );
        return;
    }
    if topics.is_empty() {
        let _ = writeln!(out, "No topics in category `{}`.", category);
        return;
    }
    let label = CATEGORY_ORDER
        .iter()
        .find(|(c, _)| *c == category)
        .map(|(_, l)| *l)
        .unwrap_or(category);
    let _ = writeln!(out, "{} — ggsql docs {} <name>", label, category);
    for topic in &topics {
        let _ = writeln!(out, "  {}", topic);
    }
}

fn cmd_skill(format: Option<DocsFormat>) {
    if !docs::SKILL.available {
        eprintln!(
            "The ggsql skill is not available in this build (network fetch failed and no cached copy was present at build time)."
        );
        std::process::exit(1);
    }

    let fmt = format.unwrap_or_else(|| {
        if std::io::stdout().is_terminal() {
            DocsFormat::Text
        } else {
            DocsFormat::Markdown
        }
    });

    match fmt {
        DocsFormat::Text => {
            let skin = termimad::MadSkin::default();
            skin.print_text(&strip_images(docs::SKILL.body));
        }
        DocsFormat::Markdown => {
            print!("{}", docs::SKILL.body);
            if !docs::SKILL.body.ends_with('\n') {
                println!();
            }
        }
        DocsFormat::Json => {
            let obj = serde_json::json!({
                "name": docs::SKILL.name,
                "description": docs::SKILL.description,
                "body": docs::SKILL.body,
            });
            match serde_json::to_string_pretty(&obj) {
                Ok(s) => println!("{}", s),
                Err(e) => {
                    eprintln!("Failed to serialize skill: {}", e);
                    std::process::exit(1);
                }
            }
        }
    }
}
