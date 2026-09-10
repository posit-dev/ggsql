/*!
Visual test harness for the documentation examples.

Every executable ```` ```{ggsql} ```` cell in the Quarto docs is a query the
project already vouches for, which makes them a ready-made corpus for
exercising a writer. This example runs them — in document order, against one
reader per source file so `CREATE TABLE` setup cells still apply — renders each
visualisation with the [`PngWriter`], and emits a single HTML report
pairing every query with its rendered output.

```sh
cargo run -p ggsql-cli --features png --example visual_test
open target/visual-test/index.html
```

Failures never abort the run: an execution error, a render error, or a panic
inside a writer is captured against its cell and the harness moves on, so one
report shows every problem in the corpus at once.

Pass `--compare` to render each query with the Vega-Lite writer as well and show
the two side by side — the comparison
[`writer/hephaestus/CLAUDE.md`](../../src/writer/hephaestus/CLAUDE.md) calls for
when checking visual correctness.
*/

use clap::Parser;
use ggsql::reader::{DuckDBReader, Reader};
use ggsql::util::escape_html;
use ggsql::validate::validate;
use ggsql::writer::{PngWriter, VegaLiteWriter, Writer};
use std::fmt::Write as _;
use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "visual_test",
    about = "Render every {ggsql} documentation example with the png writer into an HTML report"
)]
struct Args {
    /// Directories to scan for `.qmd` files, or individual `.qmd` files
    #[arg(default_values = ["doc/syntax", "doc/gallery"])]
    paths: Vec<PathBuf>,

    /// Directory to write the report and its images into
    #[arg(short, long, default_value = "target/visual-test")]
    out: PathBuf,

    /// Only include source files whose path contains this substring
    #[arg(short, long)]
    filter: Option<String>,

    /// Also render each query with the Vega-Lite writer, side by side
    #[arg(short, long)]
    compare: bool,

    /// Canvas width in pixels
    #[arg(long, default_value_t = 1500)]
    width: u32,

    /// Canvas height in pixels
    #[arg(long, default_value_t = 1000)]
    height: u32,

    /// Render resolution, which also scales the plot chrome
    #[arg(long, default_value_t = 300.0)]
    dpi: f64,
}

// ============================================================================
// The corpus: `{ggsql}` cells extracted from Quarto sources
// ============================================================================

/// One executable ggsql cell, in the order it appears in its source file.
struct Cell {
    /// 1-based position within the source file
    index: usize,
    /// Line of the opening fence, so a finding can be traced back to the source
    line: usize,
    /// Nearest preceding markdown heading, for orientation in the report
    heading: String,
    query: String,
}

/// A source file and the cells it contributes.
struct Source {
    /// Path as given on the command line, used as the report's label
    label: String,
    title: String,
    /// The file's own directory, which its cells run in
    dir: PathBuf,
    cells: Vec<Cell>,
}

/// Collect `.qmd` files from the given paths, recursing into directories.
fn collect_sources(paths: &[PathBuf], filter: Option<&str>) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for path in paths {
        walk_qmd(path, &mut found);
    }
    found.retain(|p| match filter {
        Some(f) => p.to_string_lossy().contains(f),
        None => true,
    });
    found.sort();
    found
}

fn walk_qmd(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_file() {
        if path.extension().is_some_and(|e| e == "qmd") {
            out.push(path.to_path_buf());
        }
        return;
    }
    let Ok(entries) = fs::read_dir(path) else {
        eprintln!("warning: cannot read {}", path.display());
        return;
    };
    for entry in entries.flatten() {
        walk_qmd(&entry.path(), out);
    }
}

/// Extract the executable cells from a Quarto source.
///
/// Only ```` ```{ggsql} ```` fences run; a plain ```` ```ggsql ```` fence is
/// illustrative syntax the docs deliberately do not execute.
fn parse_cells(text: &str) -> Vec<Cell> {
    let mut cells = Vec::new();
    let mut heading = String::new();
    let mut lines = text.lines().enumerate().peekable();

    while let Some((number, line)) = lines.next() {
        let trimmed = line.trim();

        if let Some(text) = heading_text(trimmed) {
            heading = text;
            continue;
        }

        if trimmed != "```{ggsql}" {
            continue;
        }

        let mut body = Vec::new();
        for (_, line) in lines.by_ref() {
            if line.trim() == "```" {
                break;
            }
            body.push(line);
        }

        // Quarto cell options (`#| code-fold: true`) lead the cell and are not SQL.
        let query = body
            .iter()
            .skip_while(|l| l.trim_start().starts_with("#|"))
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");

        if !query.trim().is_empty() {
            cells.push(Cell {
                index: cells.len() + 1,
                line: number + 1,
                heading: heading.clone(),
                query: query.trim().to_string(),
            });
        }
    }

    cells
}

/// The text of an ATX heading line, or `None` if this is not one.
fn heading_text(line: &str) -> Option<String> {
    let level = line.chars().take_while(|c| *c == '#').count();
    if level == 0 || level > 6 {
        return None;
    }
    let rest = &line[level..];
    rest.strip_prefix(' ').map(|t| t.trim().to_string())
}

/// The `title` field of the YAML front matter, if the file has one.
fn front_matter_title(text: &str) -> Option<String> {
    let body = text.strip_prefix("---\n")?;
    let end = body.find("\n---")?;
    for line in body[..end].lines() {
        if let Some(value) = line.strip_prefix("title:") {
            return Some(value.trim().trim_matches(['"', '\'']).to_string());
        }
    }
    None
}

// ============================================================================
// Running the corpus
// ============================================================================

/// What a cell turned out to be, and what came of running it.
enum Outcome {
    /// A query with a `VISUALISE` clause: the renders it produced.
    Plot {
        png: Option<String>,
        png_error: Option<String>,
        /// Vega-Lite JSON, inlined into the report when `--compare` is on
        vegalite: Option<String>,
        vegalite_error: Option<String>,
    },
    /// A cell with no visualisation — data setup, or a bare table query.
    Setup { rows: usize, columns: usize },
    /// The cell never produced a spec.
    Failed(String),
}

struct CellResult {
    cell: Cell,
    outcome: Outcome,
    warnings: Vec<String>,
    millis: u128,
}

impl CellResult {
    /// Whether anything about this cell needs a human's attention.
    fn is_problem(&self) -> bool {
        match &self.outcome {
            Outcome::Failed(_) => true,
            Outcome::Plot {
                png_error,
                vegalite_error,
                ..
            } => png_error.is_some() || vegalite_error.is_some(),
            Outcome::Setup { .. } => false,
        }
    }
}

struct SourceResult {
    label: String,
    title: String,
    cells: Vec<CellResult>,
}

/// Run every cell of one source file against a fresh reader.
///
/// The reader is per file, not per cell, because the docs rely on it: a page
/// may build a table in one cell and plot it in the next.
///
/// The cells run in their own file's directory, because a page reads its data
/// by a path relative to itself (`FROM 'minard_troops.csv'`) — that is how
/// Quarto executes them. `assets` is therefore resolved to an absolute path by
/// the caller, since it outlives that switch.
fn run_source(source: Source, args: &Args, assets: &Path) -> SourceResult {
    let restore = std::env::current_dir().ok();
    if let Err(e) = std::env::set_current_dir(&source.dir) {
        eprintln!("warning: cannot enter {}: {e}", source.dir.display());
    }

    let result = run_cells(source, args, assets);

    if let Some(dir) = restore {
        let _ = std::env::set_current_dir(dir);
    }
    result
}

fn run_cells(source: Source, args: &Args, assets: &Path) -> SourceResult {
    let Source {
        label,
        title,
        cells,
        ..
    } = source;

    let reader = match DuckDBReader::from_connection_string("duckdb://memory") {
        Ok(reader) => reader,
        Err(e) => {
            let cells = cells
                .into_iter()
                .map(|cell| CellResult {
                    cell,
                    outcome: Outcome::Failed(format!("could not open a reader: {e}")),
                    warnings: Vec::new(),
                    millis: 0,
                })
                .collect();
            return SourceResult {
                label,
                title,
                cells,
            };
        }
    };

    let png_writer = PngWriter::new(args.width, args.height, args.dpi);
    let vegalite = VegaLiteWriter::new();

    let mut results = Vec::new();
    for cell in cells {
        let start = Instant::now();
        let mut warnings = Vec::new();

        let has_spec = validate(&cell.query).map(|v| v.has_spec()).unwrap_or(true);

        let outcome = if has_spec {
            match capture(|| reader.execute(&cell.query)) {
                Err(e) => Outcome::Failed(e),
                // Known gap: this harness only renders VISUALISE cells (the
                // `Outcome` enum has no table variant), so a TABULATE cell
                // is reported as failed rather than actually rendered — fine
                // while no doc page uses TABULATE, but revisit once one does.
                Ok(spec) if spec.as_plot().is_none() => {
                    Outcome::Failed("TABULATE cells aren't rendered by this harness yet".into())
                }
                Ok(spec) => {
                    let plot = spec.as_plot().unwrap();
                    warnings.extend(plot.warnings().iter().map(|w| w.message.clone()));

                    let (png, png_error) = match capture(|| png_writer.render(&spec)) {
                        Ok(bytes) => {
                            let name = format!("{}-{:02}.png", slug(&label), cell.index);
                            match fs::write(assets.join(&name), &bytes) {
                                Ok(()) => (Some(name), None),
                                Err(e) => (None, Some(format!("could not write PNG: {e}"))),
                            }
                        }
                        Err(e) => (None, Some(e)),
                    };

                    let (vl, vl_error) = if args.compare {
                        match capture(|| vegalite.render(&spec)) {
                            Ok(json) => (Some(json), None),
                            Err(e) => (None, Some(e)),
                        }
                    } else {
                        (None, None)
                    };

                    Outcome::Plot {
                        png,
                        png_error,
                        vegalite: vl,
                        vegalite_error: vl_error,
                    }
                }
            }
        } else {
            match capture(|| reader.execute_sql(&cell.query)) {
                Ok(df) => Outcome::Setup {
                    rows: df.height(),
                    columns: df.width(),
                },
                Err(e) => Outcome::Failed(e),
            }
        };

        let millis = start.elapsed().as_millis();
        eprintln!(
            "  {}:{} cell {} — {} ({} ms)",
            label,
            cell.line,
            cell.index,
            status_word(&outcome),
            millis
        );

        results.push(CellResult {
            cell,
            outcome,
            warnings,
            millis,
        });
    }

    SourceResult {
        label,
        title,
        cells: results,
    }
}

fn status_word(outcome: &Outcome) -> &'static str {
    match outcome {
        Outcome::Failed(_) => "FAILED",
        Outcome::Setup { .. } => "setup",
        Outcome::Plot {
            png_error: Some(_), ..
        } => "RENDER FAILED",
        Outcome::Plot {
            vegalite_error: Some(_),
            ..
        } => "vega-lite failed",
        Outcome::Plot { .. } => "ok",
    }
}

/// Run a fallible step, turning both an error and a panic into a message.
///
/// A writer that panics on one cell must not take the whole corpus with it.
fn capture<T>(f: impl FnOnce() -> ggsql::Result<T>) -> Result<T, String> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(e)) => Err(e.to_string()),
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            Err(format!("panicked: {message}"))
        }
    }
}

// ============================================================================
// The report
// ============================================================================

fn slug(text: &str) -> String {
    text.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

/// Prepare a Vega-Lite spec for inlining in a `<script>` block.
///
/// The writer pretty-prints, which triples the size of a spec carrying a few
/// hundred data rows — worth compacting when 190 of them share one page. A
/// literal `</` would close the script element early, and `<\/` is a valid JSON
/// escape for the same character.
fn inline_json(text: &str) -> String {
    let compact = serde_json::from_str::<serde_json::Value>(text)
        .map(|value| value.to_string())
        .unwrap_or_else(|_| text.to_string());
    compact.replace("</", "<\\/")
}

fn write_report(results: &[SourceResult], args: &Args, out: &Path) -> std::io::Result<()> {
    let total: usize = results.iter().map(|r| r.cells.len()).sum();
    let plots = results
        .iter()
        .flat_map(|r| &r.cells)
        .filter(|c| matches!(c.outcome, Outcome::Plot { .. }))
        .count();
    let problems = results
        .iter()
        .flat_map(|r| &r.cells)
        .filter(|c| c.is_problem())
        .count();

    let aspect = format!("{} / {}", args.width, args.height);

    let mut html = String::new();
    html.push_str(&report_head(total, plots, problems, args));

    html.push_str("<nav id=\"toc\">\n");
    for result in results {
        let bad = result.cells.iter().filter(|c| c.is_problem()).count();
        let marker = if bad > 0 {
            format!(" <span class=\"count bad\">{bad}</span>")
        } else {
            String::new()
        };
        let _ = writeln!(
            html,
            "<a href=\"#{}\">{}{}</a>",
            slug(&result.label),
            escape_html(&result.label),
            marker
        );
    }
    html.push_str("</nav>\n<main>\n");

    for result in results {
        let _ = writeln!(
            html,
            "<h2 id=\"{}\" class=\"source\">{}<small>{}</small></h2>",
            slug(&result.label),
            escape_html(&result.title),
            escape_html(&result.label)
        );
        for cell in &result.cells {
            html.push_str(&render_cell(cell, &result.label, &aspect));
        }
    }

    html.push_str("</main>\n");
    html.push_str(REPORT_SCRIPT);
    if args.compare {
        html.push_str(VEGA_SCRIPT);
    }
    html.push_str("</body>\n</html>\n");

    fs::write(out.join("index.html"), html)
}

fn render_cell(cell: &CellResult, label: &str, aspect: &str) -> String {
    let mut html = String::new();
    let (badge, class) = match &cell.outcome {
        Outcome::Failed(_) => ("failed", "bad"),
        Outcome::Setup { .. } => ("setup", "neutral"),
        Outcome::Plot {
            png_error: Some(_), ..
        } => ("render failed", "bad"),
        Outcome::Plot {
            vegalite_error: Some(_),
            ..
        } => ("vega-lite failed", "warn"),
        Outcome::Plot { .. } => ("ok", "good"),
    };

    let problem = if cell.is_problem() { " problem" } else { "" };
    let _ = write!(
        html,
        "<section class=\"cell{problem}\" id=\"{}-{}\">\n\
         <header><span class=\"badge {class}\">{badge}</span>\
         <span class=\"loc\">{}:{} · cell {}</span>\
         <span class=\"heading\">{}</span>\
         <span class=\"time\">{} ms</span></header>\n",
        slug(label),
        cell.cell.index,
        escape_html(label),
        cell.cell.line,
        cell.cell.index,
        escape_html(&cell.cell.heading),
        cell.millis
    );

    let _ = write!(
        html,
        "<div class=\"body\"><pre class=\"query\"><code>{}</code></pre>\n<div class=\"renders\">",
        escape_html(&cell.cell.query)
    );

    match &cell.outcome {
        Outcome::Failed(message) => {
            let _ = write!(html, "<pre class=\"error\">{}</pre>", escape_html(message));
        }
        Outcome::Setup { rows, columns } => {
            let _ = write!(
                html,
                "<p class=\"note\">No visualisation — ran as setup, {rows} rows × {columns} columns.</p>"
            );
        }
        Outcome::Plot {
            png,
            png_error,
            vegalite,
            vegalite_error,
        } => {
            html.push_str("<figure><figcaption>png</figcaption>");
            match (png, png_error) {
                (Some(name), _) => {
                    let _ = write!(
                        html,
                        "<a href=\"assets/{name}\"><img loading=\"lazy\" src=\"assets/{name}\" alt=\"\"></a>"
                    );
                }
                (None, Some(message)) => {
                    let _ = write!(html, "<pre class=\"error\">{}</pre>", escape_html(message));
                }
                (None, None) => html.push_str("<p class=\"note\">no output</p>"),
            }
            html.push_str("</figure>");

            if let Some(json) = vegalite {
                // A ggsql Vega-Lite spec sizes itself to its container
                // (`"width": "container"`), so the pane needs a definite box
                // before anything is embedded into it — otherwise the chart
                // renders at zero height. Matching the raster canvas's aspect
                // also makes the two panes directly comparable.
                let _ = write!(
                    html,
                    "<figure><figcaption>vega-lite</figcaption>\
                     <div class=\"vl\" style=\"aspect-ratio:{aspect}\">\
                     <script type=\"application/json\">{}</script></div></figure>",
                    inline_json(json)
                );
            }
            if let Some(message) = vegalite_error {
                let _ = write!(
                    html,
                    "<figure><figcaption>vega-lite</figcaption>\
                     <pre class=\"error\">{}</pre></figure>",
                    escape_html(message)
                );
            }
        }
    }

    html.push_str("</div>");

    if !cell.warnings.is_empty() {
        html.push_str("<ul class=\"warnings\">");
        for warning in &cell.warnings {
            let _ = write!(html, "<li>{}</li>", escape_html(warning));
        }
        html.push_str("</ul>");
    }

    html.push_str("</div>\n</section>\n");
    html
}

fn report_head(total: usize, plots: usize, problems: usize, args: &Args) -> String {
    let compare = if args.compare {
        " · compared against vega-lite"
    } else {
        ""
    };
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ggsql visual test</title>
<style>{style}</style>
</head>
<body>
<header id="top">
  <h1>ggsql visual test</h1>
  <p class="summary">{total} cells · {plots} plots · <span class="{problem_class}">{problems} problems</span>
     · {width}×{height} px @ {dpi} dpi{compare}</p>
  <div class="controls">
    <input id="search" type="search" placeholder="filter by query, file or heading">
    <label><input id="only-problems" type="checkbox"> only problems</label>
  </div>
</header>
"#,
        style = STYLE,
        problem_class = if problems > 0 { "bad" } else { "good" },
        width = args.width,
        height = args.height,
        dpi = args.dpi,
    )
}

const STYLE: &str = r#"
:root { --bg:#fff; --fg:#1c1c1c; --muted:#666; --line:#e3e3e3; --good:#137333; --bad:#c5221f; --warn:#b06000; }
* { box-sizing: border-box; }
body { margin:0; font:14px/1.5 system-ui, sans-serif; color:var(--fg); background:var(--bg);
       display:grid; grid-template-columns:240px 1fr; grid-template-rows:auto 1fr; }
#top { grid-column:1/-1; padding:16px 24px; border-bottom:1px solid var(--line); position:sticky; top:0; background:var(--bg); z-index:2; }
#top h1 { margin:0 0 4px; font-size:18px; }
.summary { margin:0 0 8px; color:var(--muted); }
.controls { display:flex; gap:16px; align-items:center; }
#search { padding:4px 8px; border:1px solid var(--line); border-radius:4px; width:320px; }
#toc { padding:16px 8px 48px 16px; border-right:1px solid var(--line); overflow:auto; position:sticky; top:96px; align-self:start; max-height:calc(100vh - 96px); }
#toc a { display:block; padding:2px 4px; color:var(--fg); text-decoration:none; font-size:12px; border-radius:3px; }
#toc a:hover { background:#f2f2f2; }
main { padding:16px 24px 96px; min-width:0; }
h2.source { font-size:16px; margin:32px 0 8px; padding-top:8px; border-top:2px solid var(--line); }
h2.source small { display:block; font-weight:400; color:var(--muted); font-family:ui-monospace, monospace; }
.cell { border:1px solid var(--line); border-radius:6px; margin:12px 0; overflow:hidden; }
.cell.problem { border-color:var(--bad); }
.cell header { display:flex; gap:12px; align-items:baseline; padding:6px 10px; background:#fafafa; border-bottom:1px solid var(--line); font-size:12px; }
.cell .loc { font-family:ui-monospace, monospace; color:var(--muted); }
.cell .heading { color:var(--muted); }
.cell .time { margin-left:auto; color:var(--muted); }
.badge { font-weight:600; text-transform:uppercase; letter-spacing:.03em; font-size:10px; padding:2px 6px; border-radius:3px; background:#eee; }
.badge.good { background:#e6f4ea; color:var(--good); }
.badge.bad { background:#fce8e6; color:var(--bad); }
.badge.warn { background:#fef7e0; color:var(--warn); }
.count.bad { color:var(--bad); font-weight:600; }
.good { color:var(--good); } .bad { color:var(--bad); }
.body { display:grid; grid-template-columns:minmax(260px, 26%) 1fr; gap:16px; padding:12px; align-items:start; }
pre.query { margin:0; padding:10px; background:#f7f7f7; border-radius:4px; font:12px/1.45 ui-monospace, monospace; white-space:pre-wrap; overflow-wrap:anywhere; }
.renders { display:flex; gap:16px; flex-wrap:wrap; min-width:0; }
figure { margin:0; flex:1 1 420px; min-width:0; }
figcaption { font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin-bottom:4px; }
figure img { width:100%; height:auto; border:1px solid var(--line); border-radius:4px; background:#fff; }
.vl { width:100%; border:1px solid var(--line); border-radius:4px; overflow:hidden; }
.vl > script { display:none; }
pre.error { margin:0; padding:10px; background:#fce8e6; color:var(--bad); border-radius:4px; font:12px/1.45 ui-monospace, monospace; white-space:pre-wrap; }
.note { margin:0; color:var(--muted); }
.warnings { grid-column:1/-1; margin:0; padding:0 0 0 20px; color:var(--warn); font-size:12px; }
.hidden { display:none; }
@media (max-width:1100px) { body { grid-template-columns:1fr; } #toc { display:none; } .body { grid-template-columns:1fr; } }
"#;

const REPORT_SCRIPT: &str = r#"<script>
const cells = [...document.querySelectorAll('.cell')];
const search = document.getElementById('search');
const only = document.getElementById('only-problems');
function apply() {
  const needle = search.value.toLowerCase();
  for (const cell of cells) {
    const matches = !needle || cell.textContent.toLowerCase().includes(needle);
    const wanted = !only.checked || cell.classList.contains('problem');
    cell.classList.toggle('hidden', !(matches && wanted));
  }
}
search.addEventListener('input', apply);
only.addEventListener('change', apply);
</script>
"#;

/// Vega-Lite specs are inlined and embedded lazily, so the report works when
/// opened straight off disk and does not pay for 200 charts up front.
const VEGA_SCRIPT: &str = r#"<script src="https://cdn.jsdelivr.net/npm/vega@6"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@6"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@7"></script>
<script>
const observer = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    observer.unobserve(entry.target);
    const el = entry.target;
    const spec = JSON.parse(el.querySelector('script').textContent);
    vegaEmbed(el, spec, { actions: false })
      .catch((e) => { el.innerHTML = '<pre class="error">' + e + '</pre>'; });
  }
}, { rootMargin: '600px' });
document.querySelectorAll('.vl').forEach((el) => observer.observe(el));
</script>
"#;

// ============================================================================

fn main() {
    let args = Args::parse();

    let paths = collect_sources(&args.paths, args.filter.as_deref());
    if paths.is_empty() {
        eprintln!("No .qmd files found in {:?}", args.paths);
        std::process::exit(1);
    }

    let assets = args.out.join("assets");
    if let Err(e) = fs::create_dir_all(&assets) {
        eprintln!("Could not create {}: {e}", assets.display());
        std::process::exit(1);
    }
    // Cells run in their own page's directory, so the renders have to land at a
    // path that does not move with them.
    let assets = fs::canonicalize(&assets).unwrap_or(assets);

    let mut sources = Vec::new();
    for path in paths {
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) => {
                eprintln!("warning: cannot read {}: {e}", path.display());
                continue;
            }
        };
        let cells = parse_cells(&text);
        if cells.is_empty() {
            continue;
        }
        let label = path.to_string_lossy().replace('\\', "/");
        let title = front_matter_title(&text).unwrap_or_else(|| label.clone());
        let dir = match path.parent() {
            Some(dir) if !dir.as_os_str().is_empty() => dir.to_path_buf(),
            _ => PathBuf::from("."),
        };
        sources.push(Source {
            label,
            title,
            dir,
            cells,
        });
    }

    let total: usize = sources.iter().map(|s| s.cells.len()).sum();
    eprintln!(
        "Rendering {total} cells from {} files at {}×{} px, {} dpi\n",
        sources.len(),
        args.width,
        args.height,
        args.dpi
    );

    let start = Instant::now();
    let mut results = Vec::new();
    for source in sources {
        eprintln!("{}", source.label);
        results.push(run_source(source, &args, &assets));
    }

    if let Err(e) = write_report(&results, &args, &args.out) {
        eprintln!("Could not write the report: {e}");
        std::process::exit(1);
    }

    let problems: usize = results
        .iter()
        .flat_map(|r| &r.cells)
        .filter(|c| c.is_problem())
        .count();
    eprintln!(
        "\n{total} cells in {:.1}s — {problems} problems\nReport: {}",
        start.elapsed().as_secs_f64(),
        args.out.join("index.html").display()
    );
}
