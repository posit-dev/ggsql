/*!
The writer registry.

Every writer the CLI can drive is one [`WriterInfo`] row in [`WRITERS`]: its
name, the feature that compiles it, the words used for it in help and verbose
output, and the function that renders a [`ResolvedSpec`] with it. Dispatch, the
`--writer` long help, the `-D` long help and the "unknown writer" message are
all derived from that one list, so adding a writer means adding a row and its
render function — nothing else in the CLI changes.

The render functions return their error as a `String` rather than exiting, so
the caller decides how a failure is reported.
*/

use ggsql::reader::ResolvedSpec;
use ggsql::writer::WriterOptions;
use std::path::Path;
use std::sync::LazyLock;

// Reached only through a writer's own `check`/`render`, so a build with no
// writer at all would otherwise carry an unused import.
#[cfg(feature = "any-writer")]
use ggsql::writer::Writer;

#[cfg(feature = "html")]
use ggsql::writer::HtmlWriter;

#[cfg(feature = "vegalite")]
use ggsql::writer::VegaLiteWriter;

#[cfg(feature = "jpeg")]
use ggsql::writer::JpegWriter;

#[cfg(feature = "png")]
use ggsql::writer::PngWriter;

#[cfg(feature = "tiff")]
use ggsql::writer::TiffWriter;

#[cfg(feature = "hep")]
use ggsql::writer::HepWriter;

#[cfg(feature = "pdf")]
use ggsql::writer::PdfWriter;

#[cfg(feature = "svg")]
use ggsql::writer::SvgWriter;

#[cfg(feature = "webp")]
use ggsql::writer::WebpWriter;

/// What a writer produced: text to print, or bytes to pipe.
// A build with none of the listed writers compiled in has an unused variant;
// extend the gates when a writer of that shape is added.
pub enum Output {
    #[cfg_attr(
        not(any(feature = "vegalite", feature = "svg", feature = "html")),
        allow(dead_code)
    )]
    Text(String),
    #[cfg_attr(
        not(any(
            feature = "png",
            feature = "jpeg",
            feature = "tiff",
            feature = "webp",
            feature = "pdf",
            feature = "hep"
        )),
        allow(dead_code)
    )]
    Bin(Vec<u8>),
}

/// A render outcome: the output, plus anything the writer had to degrade to
/// produce it. Most writers report nothing; the vector formats can.
type Rendered = Result<(Output, Vec<String>), String>;

/// One writer, as the CLI sees it.
pub struct WriterInfo {
    /// The name `--writer` takes.
    pub name: &'static str,
    /// Alternative spellings accepted for `name`.
    pub aliases: &'static [&'static str],
    /// Filename extensions that imply this writer, without the dot and in
    /// lowercase. `--output`'s extension picks a writer from these when
    /// `--writer` was not given; see [`for_extension`].
    pub extensions: &'static [&'static str],
    /// The cargo feature that compiles this writer in.
    pub feature: &'static str,
    /// How the format is named in messages: "PNG", "Vega-Lite JSON".
    pub label: &'static str,
    /// One line describing the format, for `--writer`'s long help.
    pub blurb: &'static str,
    /// The `-D` settings this writer accepts, for `-D`'s long help. The
    /// writer itself remains the authority — an unknown key is its error.
    pub options: &'static str,
    /// Whether this build has the writer's feature enabled.
    pub compiled: bool,
    /// Build this writer from `options` and discard it, so a bad setting is
    /// reported before any SQL runs rather than after.
    pub check: fn(&WriterOptions) -> Result<(), String>,
    /// Render a spec with this writer.
    pub render: fn(&ResolvedSpec, &WriterOptions) -> Rendered,
}

/// Build `W` from `options` and throw it away — the whole of what a row's
/// `check` does once its feature is known to be on.
#[cfg(feature = "any-writer")]
fn check_options<W: Writer>(options: &WriterOptions) -> Result<(), String> {
    W::from_options(options)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

pub const WRITERS: &[WriterInfo] = &[
    WriterInfo {
        name: "vegalite",
        aliases: &["vl", "vega-lite"],
        extensions: &["json", "vl.json"],
        feature: "vegalite",
        label: "Vega-Lite JSON",
        blurb: "Vega-Lite specification as JSON",
        options: "none",
        compiled: cfg!(feature = "vegalite"),
        check: check_vegalite,
        render: render_vegalite,
    },
    WriterInfo {
        name: "html",
        aliases: &[],
        extensions: &["html", "htm"],
        feature: "html",
        label: "HTML table",
        blurb: "HTML <table> — the only writer that renders a TABULATE query",
        options: "none",
        compiled: cfg!(feature = "html"),
        check: check_html,
        render: render_html,
    },
    WriterInfo {
        name: "png",
        aliases: &[],
        extensions: &["png"],
        feature: "png",
        label: "PNG",
        blurb: "PNG image — lossless, alpha preserved",
        options: "width, height, units, dpi, background, compression",
        compiled: cfg!(feature = "png"),
        check: check_png,
        render: render_png,
    },
    WriterInfo {
        name: "jpeg",
        aliases: &["jpg"],
        extensions: &["jpg", "jpeg"],
        feature: "jpeg",
        label: "JPEG",
        blurb: "JPEG image — lossy; prefer png or webp for plots",
        options: "width, height, units, dpi, background, quality",
        compiled: cfg!(feature = "jpeg"),
        check: check_jpeg,
        render: render_jpeg,
    },
    WriterInfo {
        name: "tiff",
        aliases: &["tif"],
        extensions: &["tif", "tiff"],
        feature: "tiff",
        label: "TIFF",
        blurb: "TIFF image — lossless, choice of compressor",
        options: "width, height, units, dpi, background, compression",
        compiled: cfg!(feature = "tiff"),
        check: check_tiff,
        render: render_tiff,
    },
    WriterInfo {
        name: "webp",
        aliases: &[],
        extensions: &["webp"],
        feature: "webp",
        label: "WebP",
        blurb: "WebP image — lossless, and the smallest of the four",
        options: "width, height, units, dpi, background",
        compiled: cfg!(feature = "webp"),
        check: check_webp,
        render: render_webp,
    },
    WriterInfo {
        name: "svg",
        aliases: &[],
        extensions: &["svg"],
        feature: "svg",
        label: "SVG",
        blurb: "SVG vector graphic — scalable, and its text stays text",
        options: "width, height, units, dpi, background, text, embed-fonts, id-prefix",
        compiled: cfg!(feature = "svg"),
        check: check_svg,
        render: render_svg,
    },
    WriterInfo {
        name: "pdf",
        aliases: &[],
        extensions: &["pdf"],
        feature: "pdf",
        label: "PDF",
        blurb: "PDF page — vector, with the fonts subset in",
        options: "width, height, units, dpi, background, compress, links",
        compiled: cfg!(feature = "pdf"),
        check: check_pdf,
        render: render_pdf,
    },
    WriterInfo {
        name: "hep",
        aliases: &[],
        extensions: &["hep"],
        feature: "hep",
        label: "plot document",
        blurb: "Self-contained plot document, for a host that renders it itself",
        options: "width, height, units, dpi, background, lossy, embed-fonts",
        compiled: cfg!(feature = "hep"),
        check: check_hep,
        render: render_hep,
    },
];

/// The writer used when neither `--writer` nor `--output`'s extension names
/// one. `html` has no system requirements either, but only renders a
/// TABULATE query — Vega-Lite is the default because most queries VISUALISE.
pub const DEFAULT_WRITER: &str = "vegalite";

/// Closes `--writer`'s long help: the image writers all rasterise through the
/// GPU, stated once rather than in four blurbs.
const WRITER_FOOTER: &str = "png, jpeg, tiff and webp rasterise on the GPU and need a working \
                             adapter at render time. svg, pdf and hep do not.\n\n\
                             Left unset, --output's extension picks the writer \
                             (chart.pdf writes a PDF), falling back to vegalite. \
                             Set explicitly, this wins, and disagreeing with the \
                             extension is a warning rather than an error.";

/// Look up a writer by name or alias, case-insensitively.
pub fn find(name: &str) -> Option<&'static WriterInfo> {
    WRITERS.iter().find(|w| {
        w.name.eq_ignore_ascii_case(name) || w.aliases.iter().any(|a| a.eq_ignore_ascii_case(name))
    })
}

/// The writer a filename implies, from its extension.
///
/// Matches the longest extension first, so `chart.vl.json` picks Vega-Lite
/// rather than stopping at `json`, and a future two-part extension is not
/// shadowed by its own tail.
///
/// `None` for a path with no extension, an unrecognised one, or a bare `-` —
/// none of which is an error; they leave `--writer`'s default in place.
pub fn for_extension(path: &Path) -> Option<&'static WriterInfo> {
    let name = path.file_name()?.to_str()?.to_ascii_lowercase();
    // Longest first: "vl.json" before "json".
    let mut candidates: Vec<(&'static str, &'static WriterInfo)> = WRITERS
        .iter()
        .flat_map(|w| w.extensions.iter().map(move |e| (*e, w)))
        .collect();
    candidates.sort_by_key(|(e, _)| std::cmp::Reverse(e.len()));
    candidates
        .into_iter()
        .find(|(e, _)| name.len() > e.len() + 1 && name.ends_with(&format!(".{e}")))
        .map(|(_, w)| w)
}

/// The message for a `--writer` name that matches no row. Lists every writer,
/// marking the ones this build lacks — naming a real writer that isn't
/// compiled in is the commoner mistake.
pub fn unknown_writer(name: &str) -> String {
    let mut msg = format!("Unknown writer '{name}'\nAvailable writers:\n");
    for info in WRITERS {
        msg.push_str(&format!("  {:<9} {}", info.name, info.blurb));
        if !info.compiled {
            msg.push_str(&format!(" [not compiled in: --features {}]", info.feature));
        }
        msg.push('\n');
    }
    msg.pop();
    msg
}

/// The message for a writer that exists but is not in this build.
fn not_compiled(label: &str, feature: &str) -> String {
    format!("The {label} writer is not compiled in. Rebuild with --features {feature}")
}

/// The same message for a registry row, so the caller can refuse the writer
/// before running any SQL rather than after.
pub fn not_compiled_message(info: &WriterInfo) -> String {
    not_compiled(info.label, info.feature)
}

static WRITER_HELP: LazyLock<String> = LazyLock::new(|| {
    let mut help = String::from("Output format. Available writers:");
    for info in WRITERS {
        help.push_str(&format!("\n  {:<9} {}", info.name, info.blurb));
        if !info.compiled {
            help.push_str(&format!(
                " [not in this build: --features {}]",
                info.feature
            ));
        }
    }
    help.push_str("\n\n");
    help.push_str(WRITER_FOOTER);
    help
});

static OPTION_HELP: LazyLock<String> = LazyLock::new(|| {
    let mut help = String::from(
        "Settings for the chosen writer, as `key=value`. Repeatable, and one flag \
         may carry several settings separated by `;` (quote it, as most shells read \
         `;` themselves): `-D 'width=1600;dpi=150'`.\n\nSettings by writer:",
    );
    for info in WRITERS {
        help.push_str(&format!("\n  {:<9} {}", info.name, info.options));
    }
    help
});

/// `--writer`'s long help, listing every writer in the registry.
pub fn writer_help() -> String {
    WRITER_HELP.clone()
}

/// `-D`'s long help, listing each writer's settings.
pub fn option_help() -> String {
    OPTION_HELP.clone()
}

fn check_vegalite(options: &WriterOptions) -> Result<(), String> {
    #[cfg(feature = "vegalite")]
    return check_options::<VegaLiteWriter>(options);
    #[cfg(not(feature = "vegalite"))]
    {
        let _ = options;
        Ok(())
    }
}

fn render_vegalite(spec: &ResolvedSpec, options: &WriterOptions) -> Rendered {
    #[cfg(feature = "vegalite")]
    {
        let writer = VegaLiteWriter::from_options(options).map_err(|e| e.to_string())?;
        let json = writer.render(spec).map_err(|e| e.to_string())?;
        Ok((Output::Text(json), Vec::new()))
    }
    #[cfg(not(feature = "vegalite"))]
    {
        let _ = (spec, options);
        Err(not_compiled("Vega-Lite JSON", "vegalite"))
    }
}

fn check_html(options: &WriterOptions) -> Result<(), String> {
    #[cfg(feature = "html")]
    return check_options::<HtmlWriter>(options);
    #[cfg(not(feature = "html"))]
    {
        let _ = options;
        Ok(())
    }
}

fn render_html(spec: &ResolvedSpec, options: &WriterOptions) -> Rendered {
    #[cfg(feature = "html")]
    {
        let writer = HtmlWriter::from_options(options).map_err(|e| e.to_string())?;
        let html = writer.render(spec).map_err(|e| e.to_string())?;
        Ok((Output::Text(html), Vec::new()))
    }
    #[cfg(not(feature = "html"))]
    {
        let _ = (spec, options);
        Err(not_compiled("HTML table", "html"))
    }
}

fn check_png(options: &WriterOptions) -> Result<(), String> {
    #[cfg(feature = "png")]
    return check_options::<PngWriter>(options);
    #[cfg(not(feature = "png"))]
    {
        let _ = options;
        Ok(())
    }
}

fn render_png(spec: &ResolvedSpec, options: &WriterOptions) -> Rendered {
    #[cfg(feature = "png")]
    {
        let writer = PngWriter::from_options(options).map_err(|e| e.to_string())?;
        let png = writer.render(spec).map_err(|e| e.to_string())?;
        Ok((Output::Bin(png), Vec::new()))
    }
    #[cfg(not(feature = "png"))]
    {
        let _ = (spec, options);
        Err(not_compiled("PNG", "png"))
    }
}

fn check_jpeg(options: &WriterOptions) -> Result<(), String> {
    #[cfg(feature = "jpeg")]
    return check_options::<JpegWriter>(options);
    #[cfg(not(feature = "jpeg"))]
    {
        let _ = options;
        Ok(())
    }
}

fn render_jpeg(spec: &ResolvedSpec, options: &WriterOptions) -> Rendered {
    #[cfg(feature = "jpeg")]
    {
        let writer = JpegWriter::from_options(options).map_err(|e| e.to_string())?;
        let jpeg = writer.render(spec).map_err(|e| e.to_string())?;
        Ok((Output::Bin(jpeg), Vec::new()))
    }
    #[cfg(not(feature = "jpeg"))]
    {
        let _ = (spec, options);
        Err(not_compiled("JPEG", "jpeg"))
    }
}

fn check_tiff(options: &WriterOptions) -> Result<(), String> {
    #[cfg(feature = "tiff")]
    return check_options::<TiffWriter>(options);
    #[cfg(not(feature = "tiff"))]
    {
        let _ = options;
        Ok(())
    }
}

fn render_tiff(spec: &ResolvedSpec, options: &WriterOptions) -> Rendered {
    #[cfg(feature = "tiff")]
    {
        let writer = TiffWriter::from_options(options).map_err(|e| e.to_string())?;
        let tiff = writer.render(spec).map_err(|e| e.to_string())?;
        Ok((Output::Bin(tiff), Vec::new()))
    }
    #[cfg(not(feature = "tiff"))]
    {
        let _ = (spec, options);
        Err(not_compiled("TIFF", "tiff"))
    }
}

fn check_webp(options: &WriterOptions) -> Result<(), String> {
    #[cfg(feature = "webp")]
    return check_options::<WebpWriter>(options);
    #[cfg(not(feature = "webp"))]
    {
        let _ = options;
        Ok(())
    }
}

fn render_webp(spec: &ResolvedSpec, options: &WriterOptions) -> Rendered {
    #[cfg(feature = "webp")]
    {
        let writer = WebpWriter::from_options(options).map_err(|e| e.to_string())?;
        let webp = writer.render(spec).map_err(|e| e.to_string())?;
        Ok((Output::Bin(webp), Vec::new()))
    }
    #[cfg(not(feature = "webp"))]
    {
        let _ = (spec, options);
        Err(not_compiled("WebP", "webp"))
    }
}

/// The `ResolvedPlot` a Plot-only writer needs, or a clear error for a
/// TABULATE query — `svg`, `pdf` and `hep` render only plots; `html` is the
/// only writer that renders a table.
#[cfg(any(feature = "svg", feature = "pdf", feature = "hep"))]
fn require_plot(spec: &ResolvedSpec) -> Result<&ggsql::reader::ResolvedPlot, String> {
    spec.as_plot().ok_or_else(|| {
        "this writer does not support TABULATE queries; use --writer html".to_string()
    })
}

fn check_svg(options: &WriterOptions) -> Result<(), String> {
    #[cfg(feature = "svg")]
    return check_options::<SvgWriter>(options);
    #[cfg(not(feature = "svg"))]
    {
        let _ = options;
        Ok(())
    }
}

fn render_svg(spec: &ResolvedSpec, options: &WriterOptions) -> Rendered {
    #[cfg(feature = "svg")]
    {
        let writer = SvgWriter::from_options(options).map_err(|e| e.to_string())?;
        let (svg, warnings) = writer
            .render_reporting(require_plot(spec)?)
            .map_err(|e| e.to_string())?;
        Ok((Output::Text(svg), warnings))
    }
    #[cfg(not(feature = "svg"))]
    {
        let _ = (spec, options);
        Err(not_compiled("SVG", "svg"))
    }
}

fn check_pdf(options: &WriterOptions) -> Result<(), String> {
    #[cfg(feature = "pdf")]
    return check_options::<PdfWriter>(options);
    #[cfg(not(feature = "pdf"))]
    {
        let _ = options;
        Ok(())
    }
}

fn render_pdf(spec: &ResolvedSpec, options: &WriterOptions) -> Rendered {
    #[cfg(feature = "pdf")]
    {
        let writer = PdfWriter::from_options(options).map_err(|e| e.to_string())?;
        let (pdf, warnings) = writer
            .render_reporting(require_plot(spec)?)
            .map_err(|e| e.to_string())?;
        Ok((Output::Bin(pdf), warnings))
    }
    #[cfg(not(feature = "pdf"))]
    {
        let _ = (spec, options);
        Err(not_compiled("PDF", "pdf"))
    }
}

fn check_hep(options: &WriterOptions) -> Result<(), String> {
    #[cfg(feature = "hep")]
    return check_options::<HepWriter>(options);
    #[cfg(not(feature = "hep"))]
    {
        let _ = options;
        Ok(())
    }
}

fn render_hep(spec: &ResolvedSpec, options: &WriterOptions) -> Rendered {
    #[cfg(feature = "hep")]
    {
        let writer = HepWriter::from_options(options).map_err(|e| e.to_string())?;
        let (bytes, warnings) = writer
            .render_reporting(require_plot(spec)?)
            .map_err(|e| e.to_string())?;
        Ok((Output::Bin(bytes), warnings))
    }
    #[cfg(not(feature = "hep"))]
    {
        let _ = (spec, options);
        Err(not_compiled("plot document", "hep"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_writer_is_findable_by_name_and_alias() {
        for info in WRITERS {
            assert_eq!(find(info.name).map(|w| w.name), Some(info.name));
            for alias in info.aliases {
                assert_eq!(find(alias).map(|w| w.name), Some(info.name));
            }
        }
    }

    #[test]
    fn lookup_ignores_case() {
        assert_eq!(find("PNG").map(|w| w.name), Some("png"));
        assert!(find("furlongs").is_none());
    }

    #[test]
    fn every_writer_declares_at_least_one_extension() {
        for info in WRITERS {
            assert!(
                !info.extensions.is_empty(),
                "{} declares no extension, so --output could never pick it",
                info.name
            );
        }
    }

    #[test]
    fn extensions_are_lowercase_and_dotless() {
        for info in WRITERS {
            for ext in info.extensions {
                assert_eq!(*ext, ext.to_ascii_lowercase(), "{}", info.name);
                assert!(!ext.starts_with('.'), "{}: {ext}", info.name);
            }
        }
    }

    #[test]
    fn extensions_are_unique_across_writers() {
        let mut seen: Vec<&str> = WRITERS
            .iter()
            .flat_map(|w| w.extensions.iter().copied())
            .collect();
        let count = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), count, "two writers claim the same extension");
    }

    #[test]
    fn an_extension_picks_its_writer() {
        for (path, expected) in [
            ("chart.svg", "svg"),
            ("chart.pdf", "pdf"),
            ("chart.hep", "hep"),
            ("chart.png", "png"),
            ("chart.jpg", "jpeg"),
            ("chart.jpeg", "jpeg"),
            ("chart.tif", "tiff"),
            ("chart.tiff", "tiff"),
            ("chart.webp", "webp"),
            ("chart.json", "vegalite"),
            ("chart.html", "html"),
            ("chart.htm", "html"),
        ] {
            assert_eq!(
                for_extension(Path::new(path)).map(|w| w.name),
                Some(expected),
                "{path}"
            );
        }
    }

    #[test]
    fn a_two_part_extension_beats_its_own_tail() {
        // Both spellings reach vegalite, so this asserts the ordering rather
        // than the destination: the longer match wins.
        let mut candidates: Vec<&str> = WRITERS
            .iter()
            .flat_map(|w| w.extensions.iter().copied())
            .filter(|e| "chart.vl.json".ends_with(&format!(".{e}")))
            .collect();
        candidates.sort_by_key(|e| std::cmp::Reverse(e.len()));
        assert_eq!(candidates.first(), Some(&"vl.json"));
        assert_eq!(
            for_extension(Path::new("chart.vl.json")).map(|w| w.name),
            Some("vegalite")
        );
    }

    #[test]
    fn extension_matching_ignores_case() {
        assert_eq!(
            for_extension(Path::new("C.SVG")).map(|w| w.name),
            Some("svg")
        );
        assert_eq!(
            for_extension(Path::new("c.Pdf")).map(|w| w.name),
            Some("pdf")
        );
    }

    #[test]
    fn an_unrecognised_or_absent_extension_picks_nothing() {
        // None is not a failure — it leaves the default writer in place.
        for path in ["notes.txt", "chart", "-", "chart.", "archive.tar.gz"] {
            assert!(for_extension(Path::new(path)).is_none(), "{path}");
        }
    }

    #[test]
    fn a_dotfile_is_a_name_not_an_extension() {
        // `.svg` is a hidden file called "svg", not an SVG, so it picks
        // nothing rather than silently choosing a writer from a bare suffix.
        assert!(for_extension(Path::new(".svg")).is_none());
        assert!(for_extension(Path::new("dir/.pdf")).is_none());
    }

    #[test]
    fn a_directory_in_the_path_is_not_read_as_an_extension() {
        assert!(for_extension(Path::new("out.svg/chart")).is_none());
        assert_eq!(
            for_extension(Path::new("out.pdf/chart.svg")).map(|w| w.name),
            Some("svg")
        );
    }

    #[test]
    fn the_default_writer_has_a_row() {
        assert!(find(DEFAULT_WRITER).is_some());
    }

    #[test]
    fn names_and_aliases_are_unique() {
        let mut seen = Vec::new();
        for info in WRITERS {
            seen.push(info.name);
            seen.extend(info.aliases);
        }
        let mut sorted = seen.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), seen.len(), "duplicate writer name or alias");
    }

    #[test]
    fn help_mentions_every_writer() {
        let writer_help = writer_help();
        let option_help = option_help();
        let unknown = unknown_writer("nope");
        for info in WRITERS {
            assert!(writer_help.contains(info.name), "{} missing", info.name);
            assert!(option_help.contains(info.name), "{} missing", info.name);
            assert!(unknown.contains(info.name), "{} missing", info.name);
        }
    }

    #[cfg(all(feature = "html", feature = "duckdb"))]
    #[test]
    fn the_html_writer_renders_a_tabulate_query() {
        use ggsql::reader::{DuckDBReader, Reader};

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute_sql(
                "CREATE TABLE sales AS SELECT * FROM (VALUES (1, 'a'), (2, 'b')) AS t(id, name)",
            )
            .unwrap();
        let spec = reader.execute("TABULATE FROM sales").unwrap();

        let info = find("html").unwrap();
        let (output, warnings) = (info.render)(&spec, &WriterOptions::new()).unwrap();
        assert!(warnings.is_empty());
        let Output::Text(html) = output else {
            panic!("expected Output::Text from the html writer");
        };
        assert!(html.starts_with("<table>"), "{html}");
        assert!(html.contains("<th>id</th>"), "{html}");
        assert!(html.contains("<td>a</td>"), "{html}");
    }

    #[cfg(all(feature = "html", feature = "duckdb"))]
    #[test]
    fn the_html_writer_rejects_a_visualise_query() {
        use ggsql::reader::{DuckDBReader, Reader};

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let spec = reader
            .execute("SELECT 1 AS x, 2 AS y VISUALISE x, y DRAW point")
            .unwrap();

        let info = find("html").unwrap();
        assert!((info.render)(&spec, &WriterOptions::new()).is_err());
    }
}
