/*!
Visual test harness for the documentation examples.

Every executable ```` ```{ggsql} ```` cell in the Quarto docs is a query the
project already vouches for, which makes them a ready-made corpus for
exercising a writer. This example runs them — in document order, against one
reader per source file so `CREATE TABLE` setup cells still apply — renders each
visualisation, and emits a single HTML report pairing every query with its
rendered output.

```sh
cargo run -p ggsql-cli --features png --example visual_test
open target/visual-test/index.html
```

`--writer` picks what draws the corpus. `png` is the default and what a visual
check wants; `svg` needs **no GPU adapter**, so it is the mode that runs on a
headless box, and its `--baseline` comparison reads coordinates rather than
pixels. The SVG writer also reports what it could not express, which is the
regression signal
[`writer/hephaestus/CLAUDE.md`](../../src/writer/hephaestus/CLAUDE.md) says
should be empty for everything in this corpus.

```sh
cargo run -p ggsql-cli --features png --example visual_test -- --writer svg
```

Failures never abort the run: an execution error, a render error, or a panic
inside a writer is captured against its cell and the harness moves on, so one
report shows every problem in the corpus at once.

Pass `--compare` to render each query with the Vega-Lite writer as well and show
the two side by side — the comparison
[`writer/hephaestus/CLAUDE.md`](../../src/writer/hephaestus/CLAUDE.md) calls for
when checking visual correctness.
*/

use clap::{Parser, ValueEnum};
use ggsql::reader::{DuckDBReader, Reader};
use ggsql::validate::validate;
use ggsql::writer::{PngWriter, SvgWriter, VegaLiteWriter, Writer};
use std::fmt::Write as _;
use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "visual_test",
    about = "Render every {ggsql} documentation example into an HTML report"
)]
struct Args {
    /// Directories to scan for `.qmd` files, or individual `.qmd` files
    #[arg(default_values = ["doc/syntax", "doc/gallery"])]
    paths: Vec<PathBuf>,

    /// Which writer draws the corpus. `svg` needs no GPU adapter.
    #[arg(long, value_enum, default_value_t = Renderer::Png)]
    writer: Renderer,

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

    /// A previous `--out` directory to diff this run's renders against.
    ///
    /// Every cell is labelled unchanged / changed / new, with counts in the
    /// header — a renderer bump changes behaviour on purpose, so the point is to
    /// narrow the eyeballing to the cells that moved.
    #[arg(long, value_name = "DIR")]
    baseline: Option<PathBuf>,
}

/// Which writer draws the corpus.
///
/// Both go through the same composition. PNG is the picture a human eyeballs and
/// needs an adapter; SVG needs none, carries readable coordinates, and reports
/// what the format could not express.
#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Renderer {
    Png,
    Svg,
}

impl Renderer {
    /// The extension a render is written under, which is also what tells the
    /// report's `<img>` what it is looking at.
    fn extension(self) -> &'static str {
        match self {
            Renderer::Png => "png",
            Renderer::Svg => "svg",
        }
    }

    /// Draw one spec, reporting anything the format could not express.
    ///
    /// The SVG list should be empty for everything in this corpus, so it goes
    /// beside the execution warnings rather than into the render error.
    fn render(
        self,
        spec: &ggsql::reader::Spec,
        args: &Args,
    ) -> ggsql::Result<(Vec<u8>, Vec<String>)> {
        match self {
            Renderer::Png => PngWriter::new(args.width, args.height, args.dpi)
                .render(spec)
                .map(|bytes| (bytes, Vec::new())),
            Renderer::Svg => SvgWriter::new(args.width, args.height, args.dpi)
                .render_reporting(spec)
                .map(|(svg, warnings)| (svg.into_bytes(), warnings)),
        }
    }

    /// Compare a render with the baseline's, the way this format wants.
    fn compare(self, baseline: &Path, name: &str, bytes: &[u8]) -> Delta {
        match self {
            Renderer::Png => Delta::against_pixels(baseline, name, bytes),
            Renderer::Svg => Delta::against_markup(baseline, name, bytes),
        }
    }
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

/// How a render compares with the same cell in a `--baseline` run.
#[derive(Clone, Copy)]
enum Delta {
    /// Same pixels, exactly.
    Identical,
    /// Different pixels, but the whole picture matches under a small
    /// translation — the signature of a chrome-width change nudging the panel.
    /// Carries the residual left at the best alignment.
    Shifted(f64),
    /// Differs by more than a shift explains. These are the cells to look at.
    Changed(f64),
    /// The baseline has no render for this cell.
    New,
}

/// Mean absolute grey difference, 0–255, below which an aligned pair counts as
/// the same picture. Antialiasing puts a genuine match a little above zero, so
/// this cannot be `0.0`; it comes from the observed spread across the corpus.
const SHIFT_TOLERANCE: f64 = 2.0;

/// How far to search for an alignment, in pixels of the full-size render.
const MAX_SHIFT: i32 = 24;

/// Factor the comparison downsamples by before searching. The search cost is
/// quadratic in both the shift range and the resolution, and a panel shift is a
/// whole-image effect that survives a box filter.
const COMPARE_SCALE: u32 = 8;

impl Delta {
    /// Compare freshly rendered bytes with the baseline's copy of `name`.
    ///
    /// Exact equality is useless across a dependency bump: a few pixels of
    /// chrome-width change shifts every panel. Aligning first — searching a
    /// small translation for the best residual — collapses a uniform shift to
    /// nearly nothing and leaves anything structural large.
    fn against_pixels(baseline: &Path, name: &str, bytes: &[u8]) -> Self {
        let Ok(old) = fs::read(baseline.join("assets").join(name)) else {
            return Delta::New;
        };
        if old == bytes {
            return Delta::Identical;
        }
        match (Grey::decode(&old), Grey::decode(bytes)) {
            (Some(a), Some(b)) => match a.aligned_residual(&b) {
                // Different dimensions: not a shift, and not comparable.
                None => Delta::Changed(f64::INFINITY),
                Some(r) if r <= SHIFT_TOLERANCE => Delta::Shifted(r),
                Some(r) => Delta::Changed(r),
            },
            // Undecodable, so fall back to saying it moved rather than
            // claiming a match we cannot support.
            _ => Delta::Changed(f64::INFINITY),
        }
    }

    /// Compare freshly drawn markup with the baseline's copy of `name`.
    ///
    /// As a multiset of elements, not as bytes: a query with no `ORDER BY` emits
    /// the same marks in a different order each run. Sorting first leaves a real
    /// change — a moved mark, a different label, an element gained or lost —
    /// showing. A jittered layer genuinely differs each run and stays "changed".
    ///
    /// No shift alignment: markup carries its coordinates, so a moved panel
    /// shows up as changed coordinates rather than as a translated image.
    fn against_markup(baseline: &Path, name: &str, bytes: &[u8]) -> Self {
        let Ok(old) = fs::read(baseline.join("assets").join(name)) else {
            return Delta::New;
        };
        if old == bytes {
            return Delta::Identical;
        }
        match (std::str::from_utf8(&old), std::str::from_utf8(bytes)) {
            (Ok(a), Ok(b)) if sorted_elements(a) == sorted_elements(b) => Delta::Identical,
            _ => Delta::Changed(f64::INFINITY),
        }
    }

    fn label(self) -> String {
        match self {
            Delta::Identical => "identical".to_string(),
            Delta::Shifted(r) => format!("shifted · {r:.2}"),
            Delta::Changed(r) if r.is_finite() => format!("changed · {r:.2}"),
            Delta::Changed(_) => "changed".to_string(),
            Delta::New => "new".to_string(),
        }
    }

    /// CSS class, and the bucket the header counts by.
    fn class(self) -> &'static str {
        match self {
            Delta::Identical => "identical",
            Delta::Shifted(_) => "shifted",
            Delta::Changed(_) => "changed",
            Delta::New => "new",
        }
    }

    /// Whether a human still needs to look at this cell.
    fn needs_review(self) -> bool {
        matches!(self, Delta::Changed(_))
    }
}

/// An SVG's elements in a comparable order.
fn sorted_elements(svg: &str) -> Vec<&str> {
    let mut parts: Vec<&str> = svg.split('>').collect();
    parts.sort_unstable();
    parts
}

/// A render reduced to one grey byte per pixel, downsampled for comparison.
struct Grey {
    width: usize,
    height: usize,
    px: Vec<u8>,
}

impl Grey {
    /// Decode a PNG and reduce it to a downsampled grey plane.
    fn decode(bytes: &[u8]) -> Option<Self> {
        let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
        let mut reader = decoder.read_info().ok()?;
        let mut buf = vec![0; reader.output_buffer_size()?];
        let info = reader.next_frame(&mut buf).ok()?;
        let channels = match info.color_type {
            png::ColorType::Rgba => 4,
            png::ColorType::Rgb => 3,
            png::ColorType::Grayscale => 1,
            png::ColorType::GrayscaleAlpha => 2,
            png::ColorType::Indexed => return None,
        };
        // Composite onto white as a viewer would, so a transparent background
        // does not read as black and swamp the residual.
        let grey_at = |i: usize| -> f64 {
            let p = &buf[i * channels..];
            let (r, g, b, a) = match channels {
                1 => (p[0], p[0], p[0], 255),
                2 => (p[0], p[0], p[0], p[1]),
                3 => (p[0], p[1], p[2], 255),
                _ => (p[0], p[1], p[2], p[3]),
            };
            let lum = 0.299 * r as f64 + 0.587 * g as f64 + 0.114 * b as f64;
            let a = a as f64 / 255.0;
            lum * a + 255.0 * (1.0 - a)
        };

        let scale = COMPARE_SCALE as usize;
        let width = (info.width as usize).div_ceil(scale);
        let height = (info.height as usize).div_ceil(scale);
        let mut px = vec![0u8; width * height];
        for by in 0..height {
            for bx in 0..width {
                let mut sum = 0.0;
                let mut n = 0.0;
                for y in by * scale..((by + 1) * scale).min(info.height as usize) {
                    for x in bx * scale..((bx + 1) * scale).min(info.width as usize) {
                        sum += grey_at(y * info.width as usize + x);
                        n += 1.0;
                    }
                }
                px[by * width + bx] = (sum / n).round() as u8;
            }
        }
        Some(Grey { width, height, px })
    }

    /// The smallest mean absolute difference over a search of translations.
    ///
    /// `None` when the two have different dimensions, which no translation
    /// reconciles.
    fn aligned_residual(&self, other: &Grey) -> Option<f64> {
        if self.width != other.width || self.height != other.height {
            return None;
        }
        let reach = MAX_SHIFT / COMPARE_SCALE as i32;
        let mut best = f64::INFINITY;
        for dy in -reach..=reach {
            for dx in -reach..=reach {
                // Only the overlap is compared, so a shift is not penalised for
                // the sliver it moves off the canvas.
                let mut sum = 0.0f64;
                let mut n = 0usize;
                for y in 0..self.height as i32 {
                    let oy = y + dy;
                    if oy < 0 || oy >= self.height as i32 {
                        continue;
                    }
                    for x in 0..self.width as i32 {
                        let ox = x + dx;
                        if ox < 0 || ox >= self.width as i32 {
                            continue;
                        }
                        let a = self.px[y as usize * self.width + x as usize] as f64;
                        let b = other.px[oy as usize * self.width + ox as usize] as f64;
                        sum += (a - b).abs();
                        n += 1;
                    }
                }
                if n > 0 {
                    best = best.min(sum / n as f64);
                }
            }
        }
        Some(best)
    }
}

/// What a cell turned out to be, and what came of running it.
enum Outcome {
    /// A query with a `VISUALISE` clause: the renders it produced.
    Plot {
        /// Filename of the render in `assets/`, whatever writer produced it.
        image: Option<String>,
        image_error: Option<String>,
        /// How the render compares with `--baseline`, when one was given.
        delta: Option<Delta>,
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
                image_error,
                vegalite_error,
                ..
            } => image_error.is_some() || vegalite_error.is_some(),
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
/// The reader is per file, not per cell: a page may build a table in one cell
/// and plot it in the next. Cells run in their own file's directory, since a
/// page reads its data by a relative path, so the caller resolves `assets` to
/// an absolute path first.
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

    let vegalite = VegaLiteWriter::new();

    let mut results = Vec::new();
    for cell in cells {
        let start = Instant::now();
        let mut warnings = Vec::new();

        let has_visual = validate(&cell.query)
            .map(|v| v.has_visual())
            .unwrap_or(true);

        let outcome = if has_visual {
            match capture(|| reader.execute(&cell.query)) {
                Err(e) => Outcome::Failed(e),
                Ok(spec) => {
                    warnings.extend(spec.warnings().iter().map(|w| w.message.clone()));

                    let mut delta = None;
                    let (image, image_error) = match capture(|| args.writer.render(&spec, args)) {
                        Ok((bytes, degraded)) => {
                            // Beside the execution warnings, not folded into the
                            // render error: the render succeeded.
                            warnings.extend(degraded);
                            let name = format!(
                                "{}-{:02}.{}",
                                slug(&label),
                                cell.index,
                                args.writer.extension()
                            );
                            delta = args
                                .baseline
                                .as_deref()
                                .map(|b| args.writer.compare(b, &name, &bytes));
                            match fs::write(assets.join(&name), &bytes) {
                                Ok(()) => (Some(name), None),
                                Err(e) => (None, Some(format!("could not write the render: {e}"))),
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
                        image,
                        image_error,
                        delta,
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
            image_error: Some(_),
            ..
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

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Prepare a Vega-Lite spec for inlining in a `<script>` block.
///
/// The writer pretty-prints, which triples the size of a spec carrying a few
/// hundred rows — worth compacting when 190 share one page. A literal `</` would
/// close the script element early; `<\/` is a valid JSON escape for it.
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
    let bucket = |class: &str| {
        results
            .iter()
            .flat_map(|r| &r.cells)
            .filter(
                |c| matches!(&c.outcome, Outcome::Plot { delta: Some(d), .. } if d.class() == class),
            )
            .count()
    };
    let drift = args.baseline.as_deref().map(|_| {
        (
            bucket("identical") + bucket("shifted"),
            bucket("changed"),
            bucket("new"),
        )
    });

    let aspect = format!("{} / {}", args.width, args.height);

    let mut html = String::new();
    html.push_str(&report_head(total, plots, problems, drift, args));

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
            escape(&result.label),
            marker
        );
    }
    html.push_str("</nav>\n<main>\n");

    for result in results {
        let _ = writeln!(
            html,
            "<h2 id=\"{}\" class=\"source\">{}<small>{}</small></h2>",
            slug(&result.label),
            escape(&result.title),
            escape(&result.label)
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
            image_error: Some(_),
            ..
        } => ("render failed", "bad"),
        Outcome::Plot {
            vegalite_error: Some(_),
            ..
        } => ("vega-lite failed", "warn"),
        Outcome::Plot { .. } => ("ok", "good"),
    };

    let problem = if cell.is_problem() { " problem" } else { "" };
    // A shift-explained difference is not worth a human's time; a residual that
    // a shift does not explain is exactly what the toggle exists to isolate.
    let review = match &cell.outcome {
        Outcome::Plot { delta: Some(d), .. } if d.needs_review() => " review",
        _ => "",
    };
    let _ = write!(
        html,
        "<section class=\"cell{problem}{review}\" id=\"{}-{}\">\n\
         <header><span class=\"badge {class}\">{badge}</span>\
         <span class=\"loc\">{}:{} · cell {}</span>\
         <span class=\"heading\">{}</span>\
         <span class=\"time\">{} ms</span></header>\n",
        slug(label),
        cell.cell.index,
        escape(label),
        cell.cell.line,
        cell.cell.index,
        escape(&cell.cell.heading),
        cell.millis
    );

    let _ = write!(
        html,
        "<div class=\"body\"><pre class=\"query\"><code>{}</code></pre>\n<div class=\"renders\">",
        escape(&cell.cell.query)
    );

    match &cell.outcome {
        Outcome::Failed(message) => {
            let _ = write!(html, "<pre class=\"error\">{}</pre>", escape(message));
        }
        Outcome::Setup { rows, columns } => {
            let _ = write!(
                html,
                "<p class=\"note\">No visualisation — ran as setup, {rows} rows × {columns} columns.</p>"
            );
        }
        Outcome::Plot {
            image,
            image_error,
            delta,
            vegalite,
            vegalite_error,
        } => {
            let format = image
                .as_deref()
                .and_then(|name| name.rsplit_once('.'))
                .map(|(_, ext)| ext)
                .unwrap_or("render");
            let caption = match delta {
                Some(d) => format!(
                    "{format} <span class=\"delta {}\">{}</span>",
                    d.class(),
                    d.label()
                ),
                None => format.to_string(),
            };
            let _ = write!(html, "<figure><figcaption>{caption}</figcaption>");
            match (image, image_error) {
                (Some(name), _) => {
                    let _ = write!(
                        html,
                        "<a href=\"assets/{name}\"><img loading=\"lazy\" src=\"assets/{name}\" alt=\"\"></a>"
                    );
                }
                (None, Some(message)) => {
                    let _ = write!(html, "<pre class=\"error\">{}</pre>", escape(message));
                }
                (None, None) => html.push_str("<p class=\"note\">no output</p>"),
            }
            html.push_str("</figure>");

            if let Some(json) = vegalite {
                // A ggsql Vega-Lite spec sizes itself to its container, so the
                // pane needs a definite box or the chart renders at zero height.
                // Matching the raster aspect keeps the two panes comparable.
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
                    escape(message)
                );
            }
        }
    }

    html.push_str("</div>");

    if !cell.warnings.is_empty() {
        html.push_str("<ul class=\"warnings\">");
        for warning in &cell.warnings {
            let _ = write!(html, "<li>{}</li>", escape(warning));
        }
        html.push_str("</ul>");
    }

    html.push_str("</div>\n</section>\n");
    html
}

fn report_head(
    total: usize,
    plots: usize,
    problems: usize,
    drift: Option<(usize, usize, usize)>,
    args: &Args,
) -> String {
    let compare = if args.compare {
        " · compared against vega-lite"
    } else {
        ""
    };
    // `changed` is the count that matters after a dependency bump: the set a
    // shift does not explain, and therefore the set to look at.
    let drift = match drift {
        Some((aligned, changed, new)) => format!(
            " · <span class=\"delta changed\">{changed} to review</span> \
             · {aligned} same or shifted · {new} new"
        ),
        None => String::new(),
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
     · {width}×{height} px @ {dpi} dpi{compare}{drift}</p>
  <div class="controls">
    <input id="search" type="search" placeholder="filter by query, file or heading">
    <label><input id="only-problems" type="checkbox"> only problems</label>
    <label><input id="only-review" type="checkbox"> only changes to review</label>
  </div>
</header>
"#,
        style = STYLE,
        drift = drift,
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
.delta { display:inline-block; padding:0 5px; border-radius:3px; font-weight:600; letter-spacing:0; text-transform:none; }
.delta.identical { color:var(--muted); background:#f2f2f2; }
.delta.shifted { color:var(--muted); background:#f2f2f2; }
.delta.changed { color:#fff; background:var(--warn); }
.delta.new { color:#fff; background:var(--good); }
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
const review = document.getElementById('only-review');
function apply() {
  const needle = search.value.toLowerCase();
  for (const cell of cells) {
    const matches = !needle || cell.textContent.toLowerCase().includes(needle);
    const wanted = !only.checked || cell.classList.contains('problem');
    const moved = !review.checked || cell.classList.contains('review');
    cell.classList.toggle('hidden', !(matches && wanted && moved));
  }
}
search.addEventListener('input', apply);
only.addEventListener('change', apply);
review.addEventListener('change', apply);
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

    // Read from inside that same directory switch, so a relative `--baseline`
    // would silently resolve to nothing and report every cell as new.
    let mut args = args;
    if let Some(baseline) = args.baseline.take() {
        match fs::canonicalize(&baseline) {
            Ok(path) => args.baseline = Some(path),
            Err(e) => {
                eprintln!("Could not read baseline {}: {e}", baseline.display());
                std::process::exit(1);
            }
        }
    }
    let args = args;

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
