//! Renderer-backed writers.
//!
//! Every writer here renders a resolved ggsql `Spec` through the [`hephaestus`]
//! 2D scene renderer. Only the writers themselves are public; the renderer
//! behind them is an implementation detail, and this module is private.
//!
//! The work splits three ways, which is what keeps one writer per format small:
//!
//! - [`compose`] turns a `Plot` into a live `PlotComposition`. Format-independent,
//!   and where nearly all the code is.
//! - [`canvas`] carries the size, resolution and background, and parses the
//!   options they come from.
//! - [`raster`] rasterises a composition to pixels. **The only part that needs a
//!   GPU adapter** — a vector writer builds a scene from the same composition
//!   and never comes through here.
//!
//! A format's own module is then just its option parsing and one encoder call:
//! [`png`], [`jpeg`], [`tiff`], [`webp`]. What differs between them is the axis
//! each format actually has — PNG trades encode time for size, JPEG trades
//! quality for size, TIFF picks a compressor, and WebP is lossless with no rate
//! control at all — so they do not share a knob they would each have to
//! reinterpret.
//!
//! **Scope**: multi-layer plots under Cartesian, Polar, and Map projections,
//! with `FACET` faceting (Wrap/Grid, fixed + free scales); every geom except
//! `arrow`, which is a stub no writer implements; all scale types and
//! transforms, material aesthetics, plot and axis titles, and legends.
//!
//! Architecture — the abstractions and the invariants they keep — and the
//! inventory of deferred work are documented in
//! `src/writer/hephaestus/CLAUDE.md`.

mod canvas;
mod channels;
mod compose;
mod facet;
mod geom;
mod projection;
#[cfg(feature = "raster-writer")]
mod raster;
mod scales;
#[cfg(any(feature = "svg", feature = "pdf"))]
mod vector;
mod wiring;

#[cfg(feature = "hep")]
mod hep;
#[cfg(feature = "jpeg")]
mod jpeg;
#[cfg(feature = "pdf")]
mod pdf;
#[cfg(feature = "png")]
mod png;
#[cfg(feature = "svg")]
mod svg;
#[cfg(feature = "tiff")]
mod tiff;
#[cfg(feature = "webp")]
mod webp;
#[cfg(feature = "window")]
mod window;

pub use hephaestus::color::{rgba, Color};

pub use canvas::Canvas;
#[cfg(feature = "hep")]
use canvas::CANVAS_SIZE_OPTIONS;

#[cfg(feature = "raster-writer")]
pub use raster::{RasterRenderer, MAX_RASTER_DIMENSION};

#[cfg(feature = "hep")]
pub use hep::HepWriter;
#[cfg(feature = "jpeg")]
pub use jpeg::JpegWriter;
#[cfg(feature = "pdf")]
pub use pdf::PdfWriter;
#[cfg(feature = "png")]
pub use png::PngWriter;
#[cfg(feature = "svg")]
pub use svg::SvgWriter;
#[cfg(feature = "tiff")]
pub use tiff::{TiffCompression, TiffWriter};
#[cfg(feature = "webp")]
pub use webp::WebpWriter;
#[cfg(feature = "window")]
pub use window::PlotViewer;

// Re-exported so a caller can name the setting without depending on the
// renderer crate; the variants are the format's own vocabulary.
#[cfg(feature = "png")]
pub use hephaestus::png::PngCompression;

// The shared corpus: every `renders_*` test is one query the composition layer
// must handle, driven through every writer this build has. The vector writers
// need no GPU adapter, so they run in CI; the raster assertion skips without one.
#[cfg(all(
    test,
    feature = "duckdb",
    any(feature = "png", feature = "svg", feature = "pdf")
))]
mod tests {
    use super::*;
    use crate::reader::{DuckDBReader, Reader};
    // Only the raster branch of `assert_renders` calls a trait method; the
    // vector writers report through their own inherent `render_reporting`.
    #[cfg(feature = "png")]
    use crate::writer::Writer;
    use crate::GgsqlError;
    #[cfg(feature = "png")]
    use crate::Result;
    use hephaestus::scales::chrome::AxisSide;

    /// The canvas every corpus render uses. Small, since none of these tests
    /// look at the picture — only that the whole pipeline ran.
    const CORPUS_SIZE: (u32, u32, f64) = (640, 480, 96.0);

    fn spec_for(query: &str) -> crate::reader::Spec {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader.execute(query).unwrap()
    }

    /// Render `query` through every compiled writer, asserting each output
    /// carries its own format's signature and that nothing was degraded.
    ///
    /// The vector assertions never skip, so they keep proving the composition
    /// built and drew on a machine where the raster one can't. Warnings must be
    /// empty: ggsql draws nothing a vector format cannot express.
    fn assert_renders(query: &str) {
        let (w, h, dpi) = CORPUS_SIZE;
        let spec = spec_for(query);

        #[cfg(feature = "svg")]
        {
            let (svg, warnings) = SvgWriter::new(w, h, dpi)
                .render_reporting(&spec)
                .unwrap_or_else(|e| panic!("svg render failed: {e}"));
            assert!(svg.starts_with("<svg"), "svg output should be an <svg>");
            assert!(
                svg.contains("</svg>"),
                "svg output should be closed: {}",
                &svg[..svg.len().min(200)]
            );
            assert!(
                svg.matches("<path").count() > 0,
                "an svg with no <path> drew nothing"
            );
            assert!(warnings.is_empty(), "svg degraded the plot: {warnings:?}");
        }

        #[cfg(feature = "pdf")]
        {
            let (pdf, warnings) = PdfWriter::new(w, h, dpi)
                .render_reporting(&spec)
                .unwrap_or_else(|e| panic!("pdf render failed: {e}"));
            assert!(pdf.starts_with(b"%PDF-"), "pdf output should be a PDF");
            assert!(
                pdf.ends_with(b"%%EOF\n") || pdf.ends_with(b"%%EOF"),
                "pdf output should be terminated"
            );
            assert!(warnings.is_empty(), "pdf degraded the plot: {warnings:?}");
        }

        // Last, and the only one that tolerates a headless box.
        #[cfg(feature = "png")]
        assert_png_or_skip(PngWriter::new(w, h, dpi).render(&spec));
    }

    /// The panels' `(top, right)` strip labels, in panel order. Exercises the
    /// facet layout and labelling without rendering, so it needs no GPU.
    fn strips(query: &str) -> Vec<(Option<String>, Option<String>)> {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let spec = reader.execute(query).unwrap();
        let (_, panels) = facet::build_panels(spec.plot(), spec.data()).unwrap();
        panels
            .iter()
            .map(|p| (p.strip_top.clone(), p.strip_right.clone()))
            .collect()
    }

    /// The figure's composition-level axis titles. Needs no GPU.
    fn axis_titles(query: &str) -> Vec<(AxisSide, String)> {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let spec = reader.execute(query).unwrap();
        projection::composition_axis_titles(spec.plot())
    }

    /// Just the top strip labels, in panel order.
    fn top_strips(query: &str) -> Vec<String> {
        strips(query)
            .into_iter()
            .map(|(top, _)| top.unwrap_or_default())
            .collect()
    }

    /// Assert a PNG was produced, tolerating headless CI with no GPU adapter.
    #[cfg(feature = "png")]
    fn assert_png_or_skip(result: Result<Vec<u8>>) {
        match result {
            Ok(png) => assert!(
                png.starts_with(&[0x89, b'P', b'N', b'G']),
                "output should carry the PNG signature"
            ),
            Err(GgsqlError::WriterError(msg)) if msg.contains("GPU renderer") => {
                eprintln!("skipping render assertion: {msg}");
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn renders_basic_point_plot() {
        assert_renders(
            "SELECT 1 AS x, 2 AS y UNION ALL SELECT 2, 3 UNION ALL SELECT 3, 1 \
             VISUALISE x AS x, y AS y DRAW point",
        );
    }

    #[test]
    fn renders_categorical_color_with_legend() {
        assert_renders(
            "SELECT 1 AS x, 2 AS y, 'a' AS grp UNION ALL SELECT 2, 3, 'b' \
             UNION ALL SELECT 3, 1, 'a' \
             VISUALISE x AS x, y AS y, grp AS color DRAW point",
        );
    }

    #[test]
    fn renders_continuous_size() {
        assert_renders(
            "SELECT 1 AS x, 2 AS y, 10 AS w UNION ALL SELECT 2, 3, 40 \
             UNION ALL SELECT 3, 1, 90 \
             VISUALISE x AS x, y AS y, w AS size DRAW point",
        );
    }

    #[test]
    fn renders_shape_legend() {
        // A non-color legend key must be given a color to paint, else the
        // swatches come out empty next to their labels.
        assert_renders(
            "SELECT x, y, g FROM (VALUES (1,2,'a'),(2,3,'b'),(3,1,'c')) t(x,y,g) \
             VISUALISE x AS x, y AS y, g AS shape DRAW point",
        );
    }

    #[test]
    fn renders_linetype_legend() {
        assert_renders(
            "SELECT x, y, g FROM (VALUES (1,2,'a'),(2,3,'a'),(1,1,'b'),(2,2,'b')) t(x,y,g) \
             VISUALISE x AS x, y AS y, g AS linetype DRAW line",
        );
    }

    /// An identity `linetype` column holds ggsql names or hex patterns, so it must
    /// go through `map_linetype` like a literal — the channel takes dash patterns.
    #[test]
    fn renders_identity_linetype() {
        assert_renders(
            "SELECT x, y, lt FROM (VALUES (1,2,'dashed'),(2,3,'dashed'),(1,1,'dotted'),(2,2,'dotted')) t(x,y,lt) \
             VISUALISE x AS x, y AS y, lt AS linetype DRAW line SCALE IDENTITY linetype",);
    }

    #[test]
    fn renders_colorbar_beside_size_legend() {
        // A merged colorbar for `color` beside a keyed size legend whose glyphs
        // fall back to a neutral color (`fill` holds domain values, not a constant).
        assert_renders(
            "SELECT x, y, c, w FROM (VALUES (1,2,10,100),(2,3,50,200),(3,1,90,300)) t(x,y,c,w) \
             VISUALISE x AS x, y AS y, c AS color, w AS size DRAW point",
        );
    }

    #[test]
    fn renders_log_scale() {
        assert_renders(
            "SELECT 1 AS x, 2 AS y UNION ALL SELECT 10, 3 UNION ALL SELECT 100, 1 \
             VISUALISE x AS x, y AS y DRAW point SCALE x VIA log",
        );
    }

    #[test]
    fn renders_grouped_line() {
        assert_renders(
            "SELECT 1 AS x, 2 AS y, 'a' AS g UNION ALL SELECT 2, 3, 'a' \
             UNION ALL SELECT 1, 1, 'b' UNION ALL SELECT 2, 2, 'b' \
             VISUALISE x AS x, y AS y, g AS color DRAW line",
        );
    }

    #[test]
    fn renders_bar() {
        assert_renders(
            "SELECT 'a' AS cat, 3 AS v UNION ALL SELECT 'b', 5 UNION ALL SELECT 'c', 2 \
             VISUALISE cat AS x, v AS y DRAW bar",
        );
    }

    #[test]
    fn renders_dodged_bar() {
        assert_renders(
            "SELECT x, grp, v FROM (VALUES ('a','p',3),('a','q',5),('b','p',2),('b','q',4)) \
             t(x, grp, v) \
             VISUALISE x AS x, v AS y, grp AS fill DRAW bar SETTING position => 'dodge'",
        );
    }

    #[test]
    fn renders_histogram() {
        assert_renders(
            "SELECT x FROM (VALUES (1),(2),(2),(3),(3),(3),(4),(4),(5)) t(x) \
             VISUALISE x AS x DRAW histogram",
        );
    }

    #[test]
    fn renders_area() {
        assert_renders(
            "SELECT 1 AS x, 2 AS y UNION ALL SELECT 2, 4 UNION ALL SELECT 3, 3 \
             VISUALISE x AS x, y AS y DRAW area",
        );
    }

    #[test]
    fn renders_ribbon() {
        assert_renders(
            "SELECT 1 AS x, 1 AS lo, 3 AS hi UNION ALL SELECT 2, 2, 5 \
             UNION ALL SELECT 3, 1, 4 \
             VISUALISE x AS x, lo AS ymin, hi AS ymax DRAW ribbon",
        );
    }

    #[test]
    fn renders_segment() {
        assert_renders(
            "SELECT 0 AS x, 0 AS y, 1 AS xend, 2 AS yend UNION ALL SELECT 1, 1, 2, 0 \
             VISUALISE x AS x, y AS y, xend AS xend, yend AS yend DRAW segment",
        );
    }

    #[test]
    fn renders_text() {
        assert_renders(
            "SELECT 1 AS x, 2 AS y, 'hi' AS lab UNION ALL SELECT 2, 3, 'there' \
             VISUALISE x AS x, y AS y, lab AS label DRAW text",
        );
    }

    #[test]
    fn renders_text_styled() {
        assert_renders(
            "SELECT 1 AS x, 1 AS y, 'a' AS lab UNION ALL SELECT 2, 2, 'Hello' \
             UNION ALL SELECT 3, 3, 'z' \
             VISUALISE x AS x, y AS y, lab AS label, 30 AS rotation, \
             'bold' AS fontweight, 22 AS fontsize DRAW text",
        );
    }

    /// A scaled `fontsize` on a layer whose face is set: the legend key is
    /// dressed from the same material table the glyphs are, so `family` /
    /// `weight` / `italic` / `angle` all have to reach it.
    #[test]
    fn renders_text_font_legend() {
        assert_renders(
            "SELECT 1 AS x, 1 AS y, 'a' AS lab, 10 AS sz UNION ALL SELECT 2, 2, 'b', 20 \
             UNION ALL SELECT 3, 3, 'c', 30 \
             VISUALISE x AS x, y AS y, lab AS label, sz AS fontsize \
             DRAW text SETTING typeface => 'Times New Roman', fontweight => 'bold', \
             italic => true, rotation => 20 SCALE fontsize TO (10, 30)",
        );
    }

    /// A label carrying markdown: `parse` defaults on, so the row goes through
    /// hephaestus's rich-text shaper rather than being drawn with its markers.
    #[test]
    fn renders_text_markdown() {
        assert_renders(
            "SELECT 1 AS x, 1 AS y, '**bold** and {.red red}' AS lab \
             UNION ALL SELECT 2, 2, '`code` and ~~strike~~' \
             VISUALISE x AS x, y AS y, lab AS label DRAW text",
        );
    }

    /// `SETTING parse => false` opts the layer out, drawing the markers literally.
    #[test]
    fn renders_text_markdown_off() {
        assert_renders(
            "SELECT 1 AS x, 1 AS y, '**bold** and {.red red}' AS lab \
             VISUALISE x AS x, y AS y, lab AS label DRAW text SETTING parse => false",
        );
    }

    /// The glyph outline survives the markdown path: hephaestus folds the row's
    /// `text_stroke` onto the rich sheet's root selector rather than dropping it.
    #[test]
    fn renders_text_markdown_with_stroke() {
        assert_renders(
            "SELECT 1 AS x, 1 AS y, '**bold**' AS lab \
             VISUALISE x AS x, y AS y, lab AS label \
             DRAW text SETTING fontsize => 30, stroke => 'red', rotation => 20",
        );
    }

    /// Markdown chrome: a `LABEL` string is rich text too, so the title, subtitle,
    /// caption and axis titles all shape through the rich pipeline.
    #[test]
    fn renders_markdown_chrome() {
        assert_renders(
            "SELECT 1 AS x, 2 AS y UNION ALL SELECT 2, 3 \
             VISUALISE x AS x, y AS y DRAW point \
             LABEL title => 'A **bold** title', subtitle => '{.red red} subtitle', \
             caption => '*italic* caption', x => 'axis *italic*'",
        );
    }

    /// The same aesthetics as *columns*, which take the identity path rather than
    /// the literal one: strings, booleans and degrees, each converted per row.
    #[test]
    fn renders_text_mapped_font() {
        assert_renders(
            "SELECT 1 AS x, 1 AS y, 'a' AS lab, 'Times New Roman' AS face, 'bold' AS wt, \
             true AS it, 0 AS rot \
             UNION ALL SELECT 2, 2, 'b', 'Helvetica', 'light', false, 45 \
             VISUALISE x AS x, y AS y, lab AS label, face AS typeface, wt AS fontweight, \
             it AS italic, rot AS rotation DRAW text",
        );
    }

    #[test]
    fn renders_polygon() {
        assert_renders(
            "SELECT x, y FROM (VALUES (0,0),(2,0),(1,2)) t(x, y) \
             VISUALISE x AS x, y AS y DRAW polygon",
        );
    }

    #[test]
    fn renders_boxplot() {
        assert_renders(
            "SELECT g, y FROM (VALUES ('a',1),('a',5),('a',3),('a',9),('a',2),('a',20), \
             ('b',4),('b',6),('b',5),('b',7),('b',3)) t(g, y) \
             VISUALISE g AS x, y AS y DRAW boxplot",
        );
    }

    #[test]
    fn renders_boxplot_fill_by_group() {
        assert_renders(
            "SELECT g, y FROM (VALUES ('a',1),('a',5),('a',3),('a',9),('a',2), \
             ('b',4),('b',6),('b',5),('b',7),('b',3)) t(g, y) \
             VISUALISE g AS x, y AS y, g AS fill DRAW boxplot",
        );
    }

    #[test]
    fn renders_diagonal_rule() {
        assert_renders(
            "SELECT 0 AS i VISUALISE i AS y DRAW rule \
             SETTING slope => 1 SCALE x FROM (0, 10) SCALE y FROM (0, 10)",
        );
        // The dash pattern is honored on the computed segment.
        assert_renders(
            "SELECT 0 AS i VISUALISE i AS y DRAW rule \
             SETTING slope => 1, linetype => 'dashed', linewidth => 2 \
             SCALE x FROM (0, 10) SCALE y FROM (0, 10)",
        );
        // One line per row: three intercepts → three parallel lines.
        assert_renders(
            "SELECT * FROM (VALUES (0),(2),(4)) t(i) VISUALISE i AS y DRAW rule \
             SETTING slope => 1 SCALE x FROM (0, 10) SCALE y FROM (0, 15)",
        );
    }

    #[test]
    fn renders_multiple_diagonal_rules() {
        // Per-row slope + intercept + a mapped color: three differently-sloped
        // ablines over a scatter.
        assert_renders(
            "WITH points AS (SELECT * FROM (VALUES (0, 5), (5, 15), (10, 25)) t(x, y)), \
                  lines AS (SELECT * FROM (VALUES (2, 5, 'A'), (1, 10, 'B'), (3, 0, 'C')) \
                            t(slope, y, line_id)) \
             SELECT * FROM points VISUALISE \
             DRAW point MAPPING x AS x, y AS y \
             DRAW rule MAPPING slope AS slope, y AS y, line_id AS color FROM lines",
        );
    }

    #[test]
    fn renders_constant_aesthetics() {
        // Constant material values from `SETTING` arrive as `AestheticValue::Literal`
        // and must be honored (color/size on points, linetype/linewidth on a line).
        assert_renders(
            "SELECT * FROM (VALUES (1,1),(2,3),(3,2)) t(a,b) \
             VISUALISE a AS x, b AS y DRAW point SETTING color => 'red', size => 8",
        );
        assert_renders(
            "SELECT * FROM (VALUES (1,1),(2,3),(3,2)) t(a,b) \
             VISUALISE a AS x, b AS y DRAW line \
             SETTING color => 'steelblue', linetype => 'dashed', linewidth => 2",
        );
    }

    #[test]
    fn renders_multilayer_point_line() {
        // Two layers share one pair of axes / position scales.
        assert_renders(
            "SELECT * FROM (VALUES (1,2),(2,4),(3,5),(4,4),(5,7)) t(a,b) \
             VISUALISE a AS x, b AS y DRAW point DRAW line",
        );
    }

    #[test]
    fn renders_multilayer_overlay() {
        // Bar + point overlay (point drawn over bar) over a shared discrete x.
        assert_renders(
            "SELECT g, b FROM (VALUES ('a',2),('b',4),('c',5),('d',3)) t(g,b) \
             VISUALISE g AS x, b AS y DRAW bar DRAW point SETTING color => 'red'",
        );
    }

    #[test]
    fn renders_multilayer_abline() {
        // A diagonal reference line overlaid on a scatter spans the shared
        // resolved x/y domain.
        assert_renders(
            "SELECT * FROM (VALUES (1,2),(2,4),(3,5),(4,4),(5,7)) t(a,b) \
             VISUALISE a AS x, b AS y DRAW point PLACE rule SETTING slope => 1, y => 0",
        );
    }

    #[test]
    fn renders_multilayer_shared_legend() {
        // Two layers both colored by the same variable → one collapsed legend.
        assert_renders(
            "SELECT g, a, b FROM (VALUES ('p',1,2),('p',2,4),('q',3,5),('q',4,4)) t(g,a,b) \
             VISUALISE a AS x, b AS y, g AS color DRAW point DRAW line",
        );
    }

    #[test]
    fn renders_boxplot_styled() {
        assert_renders(
            "SELECT g, y FROM (VALUES ('a',1),('a',5),('a',3),('a',9),('a',2), \
             ('b',4),('b',6),('b',5),('b',7),('b',3)) t(g, y) \
             VISUALISE g AS x, y AS y, 'navy' AS stroke DRAW boxplot",
        );
    }

    #[test]
    fn renders_boxplot_stroke_by_group() {
        // Data-mapped stroke colors every component (box/whisker/median/outlier)
        // per group and registers one collapsed legend.
        assert_renders(
            "SELECT g, y FROM (VALUES ('a',1),('a',5),('a',3),('a',9),('a',2),('a',40), \
             ('b',4),('b',6),('b',5),('b',7),('b',3)) t(g, y) \
             VISUALISE g AS x, y AS y, g AS stroke DRAW boxplot",
        );
    }

    #[test]
    fn renders_tile_sized() {
        // `width`/`height` settings shrink discrete tiles within their band.
        assert_renders(
            "SELECT a, b, v FROM (VALUES ('x','p',1),('y','q',2),('x','q',3),('y','p',4)) t(a,b,v) \
             VISUALISE a AS x, b AS y, v AS fill DRAW tile SETTING width => 0.5, height => 0.5",);
    }

    #[test]
    fn renders_tile_mixed_discrete_and_continuous_axes() {
        // The tile stat parameterises each direction on its own, so a tile can be
        // banded on one axis and spanned by extents on the other.
        let data =
            "SELECT c, n, v FROM (VALUES ('x',1.0,1),('y',2.0,2),('x',2.0,3),('y',1.0,4)) t(c,n,v)";
        assert_renders(&format!(
            "{data} VISUALISE c AS x, n AS y, v AS fill DRAW tile"
        ));
        assert_renders(&format!(
            "{data} VISUALISE n AS x, c AS y, v AS fill DRAW tile"
        ));
    }

    #[test]
    fn renders_text_keyword_justification_column() {
        // A `vjust` column of keywords is read as keywords: casting it to numbers
        // first would silently make every anchor NaN.
        assert_renders(
            "SELECT x, y, l, j FROM (VALUES (1,1,'one','top'),(2,2,'two','bottom')) t(x,y,l,j) \
             VISUALISE x AS x, y AS y, l AS label, j AS vjust DRAW text",
        );
    }

    #[test]
    fn renders_violin() {
        assert_renders(
            "SELECT g, y FROM (VALUES ('a',1),('a',5),('a',3),('a',9),('a',2), \
             ('b',4),('b',6),('b',5),('b',7),('b',3)) t(g, y) \
             VISUALISE g AS x, y AS y DRAW violin",
        );
    }

    #[test]
    fn renders_polar_pie() {
        // A stacked bar under polar becomes a pie: pos2 (count) → theta, pos1
        // (dummy) → radius. The 180° slice exercises the wide-wedge path.
        assert_renders(
            "SELECT c FROM (VALUES ('a'),('a'),('a'),('b'),('b'),('c')) t(c) \
             VISUALISE c AS fill DRAW bar PROJECT TO polar",
        );
    }

    #[test]
    fn renders_polar_donut() {
        // `inner` opens a centre hole (donut).
        assert_renders(
            "SELECT c FROM (VALUES ('a'),('a'),('a'),('b'),('b'),('c')) t(c) \
             VISUALISE c AS fill DRAW bar PROJECT TO polar SETTING inner => 0.5",
        );
    }

    #[test]
    fn renders_wrap_facet() {
        assert_renders(
            "SELECT 1 AS x, 2 AS y, 'a' AS g UNION ALL SELECT 2, 3, 'b' \
             UNION ALL SELECT 3, 1, 'a' UNION ALL SELECT 4, 5, 'c' \
             VISUALISE x AS x, y AS y DRAW point FACET g",
        );
    }

    #[test]
    fn renders_grid_facet() {
        assert_renders(
            "SELECT 1 AS x, 2 AS y, 'a' AS r, 'p' AS c UNION ALL SELECT 2, 3, 'b', 'p' \
             UNION ALL SELECT 3, 1, 'a', 'q' UNION ALL SELECT 4, 5, 'b', 'q' \
             VISUALISE x AS x, y AS y DRAW point FACET r BY c",
        );
    }

    #[test]
    fn renders_sparse_grid_facet() {
        // `('b','q')` has no rows, but the cell is still framed, gridded, axed and
        // strip-labelled so the grid stays rectangular.
        assert_renders(
            "SELECT 1 AS x, 2 AS y, 'a' AS r, 'p' AS c UNION ALL SELECT 2, 3, 'b', 'p' \
             UNION ALL SELECT 3, 1, 'a', 'q' \
             VISUALISE x AS x, y AS y DRAW point FACET r BY c",
        );
    }

    #[test]
    fn renders_sparse_grid_facet_free() {
        // An empty cell has no extent of its own, so a free dimension falls back to
        // the shared scale there — the axis and channel bindings must still resolve.
        assert_renders(
            "SELECT 1 AS x, 2 AS y, 'a' AS r, 'p' AS c UNION ALL SELECT 2, 3, 'b', 'p' \
             UNION ALL SELECT 3, 1, 'a', 'q' \
             VISUALISE x AS x, y AS y DRAW point FACET r BY c SETTING free => ['x','y']",
        );
    }

    #[test]
    fn renders_faceted_bar_with_color() {
        assert_renders(
            "SELECT g, k FROM (VALUES ('a','x'),('a','y'),('b','x'),('b','y'),('a','x')) t(g, k) \
             VISUALISE k AS x, k AS fill DRAW bar FACET g",
        );
    }

    #[test]
    fn renders_free_scale_facet() {
        // Panels with very different data ranges: free scales give each panel its
        // own per-panel domain and axes.
        assert_renders(
            "SELECT x, y, g FROM (VALUES (1,1,'a'),(2,2,'a'),(3,3,'a'),\
             (100,100,'b'),(200,200,'b'),(300,300,'b')) t(x,y,g) \
             VISUALISE x AS x, y AS y DRAW point FACET g SETTING free => ['x','y']",
        );
    }

    #[test]
    fn renders_polar_facet() {
        // A pie per panel, sharing the fill scale; proportions differ per panel.
        assert_renders(
            "SELECT c, panel FROM (VALUES \
             ('a','one'),('a','one'),('b','one'),('c','one'),\
             ('a','two'),('b','two'),('b','two'),('b','two'),('c','two')) t(c, panel) \
             VISUALISE c AS fill DRAW bar PROJECT TO polar FACET panel",
        );
    }

    #[cfg(feature = "spatial")]
    #[test]
    fn renders_spatial() {
        // A bare `spatial` geom (no PROJECT): two polygons filled by a value,
        // framed to the geometry bbox under Cartesian with equal aspect.
        assert_renders(
            "INSTALL spatial; LOAD spatial; \
             SELECT ST_GeomFromText('POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))') AS geom, \
             200 AS population \
             UNION ALL SELECT ST_GeomFromText('POLYGON ((1 0, 2 0, 2 1, 1 1, 1 0))'), 150 \
             VISUALISE DRAW spatial MAPPING population AS fill",
        );
    }

    #[cfg(feature = "spatial")]
    #[test]
    fn renders_spatial_mapped_opacity() {
        // A data-mapped scalar aesthetic (opacity) must vary per feature and
        // register a legend, not collapse to a constant.
        assert_renders(
            "INSTALL spatial; LOAD spatial; \
             SELECT ST_GeomFromText('POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))') AS geom, \
             10 AS v \
             UNION ALL SELECT ST_GeomFromText('POLYGON ((1 0, 2 0, 2 1, 1 1, 1 0))'), 90 \
             VISUALISE DRAW spatial MAPPING v AS opacity",
        );
    }

    #[cfg(feature = "spatial")]
    #[test]
    fn renders_map() {
        // A projected world map: pre-projected geometry + Custom projection
        // boundary + graticules from `computed`.
        assert_renders("VISUALISE FROM ggsql:world DRAW spatial PROJECT TO orthographic");
    }

    /// Under a map `PROJECT`, ggsql expands these layers into per-vertex rows and
    /// remaps the extent aesthetics onto `pos1`/`pos2`, so each must draw as a
    /// polyline or a polygon rather than as its usual mark — otherwise a segment
    /// is zero-length, a ribbon zero-height, a rule a fan of straight lines, and
    /// a tile a box per vertex.
    #[cfg(feature = "spatial")]
    #[test]
    fn renders_densified_segment() {
        assert_renders(
            "INSTALL spatial; LOAD spatial; \
             SELECT * FROM (VALUES (-100,30,20,60),(-50,-20,100,10)) t(x1,y1,x2,y2) \
             VISUALISE x1 AS x, y1 AS y, x2 AS xend, y2 AS yend DRAW segment \
             SETTING stroke => 'firebrick', linewidth => 2 PROJECT x, y TO robinson",
        );
    }

    #[cfg(feature = "spatial")]
    #[test]
    fn renders_densified_ribbon() {
        assert_renders(
            "INSTALL spatial; LOAD spatial; \
             SELECT * FROM (VALUES (-160,-20,20),(-80,0,40),(0,10,50),(80,-10,30)) t(x,lo,hi) \
             VISUALISE x AS x, lo AS ymin, hi AS ymax DRAW ribbon \
             SETTING fill => 'steelblue' PROJECT x, y TO robinson",
        );
    }

    #[cfg(feature = "spatial")]
    #[test]
    fn renders_densified_rule() {
        // A rule spans the clip bbox, so its meridians curve with the projection.
        assert_renders(
            "INSTALL spatial; LOAD spatial; \
             SELECT * FROM (VALUES (-100),(0),(100)) t(x) VISUALISE x AS x DRAW rule \
             SETTING stroke => 'darkgreen', linetype => 'dashed' PROJECT x, y TO robinson",
        );
    }

    #[cfg(feature = "spatial")]
    #[test]
    fn renders_densified_tile() {
        assert_renders(
            "INSTALL spatial; LOAD spatial; \
             SELECT * FROM (VALUES (-120,-30,5),(-40,20,9),(40,-10,3)) t(x,y,v) \
             VISUALISE x AS x, y AS y, v AS fill DRAW tile \
             SETTING width => 40, height => 30 PROJECT x, y TO robinson",
        );
    }

    #[cfg(feature = "spatial")]
    #[test]
    fn renders_map_over_spatial_base() {
        // A non-spatial layer over a spatial base map: both must frame to ggsql's
        // bbox so the segments land on the boundary, not on their own extent.
        assert_renders(
            "WITH routes AS (SELECT * FROM (VALUES (-74,40,2,48,'a'),(151,-34,18,-34,'b')) \
             t(x1,y1,x2,y2,route)) \
             VISUALISE \
             DRAW spatial MAPPING * FROM ggsql:world \
             DRAW segment MAPPING x1 AS x, y1 AS y, x2 AS xend, y2 AS yend, route AS stroke \
             FROM routes \
             PROJECT x, y TO robinson",
        );
    }

    /// A 6-row fixture whose `g` is categorical and `v` numeric.
    const FACET_DATA: &str = "SELECT g, v, y FROM (VALUES \
         ('a',5,1),('a',7,2),('b',15,3),('b',18,1),('c',25,2),('c',28,3)) t(g,v,y)";

    #[test]
    fn axis_titles_are_one_per_dimension() {
        // Axis titles are outer chrome: one per dimension for the whole figure,
        // however many panels, free or not.
        let expected = vec![
            (AxisSide::Bottom, "v".to_string()),
            (AxisSide::Left, "y".to_string()),
        ];
        for facet in [
            "",
            "FACET g",
            "FACET g SETTING free => ('x', 'y')",
            "FACET g BY y",
        ] {
            let query = format!("{FACET_DATA} VISUALISE v AS x, y AS y DRAW point {facet}");
            assert_eq!(axis_titles(&query), expected, "{facet}");
        }
    }

    #[test]
    fn axis_titles_follow_labels() {
        assert_eq!(
            axis_titles(&format!(
                "{FACET_DATA} VISUALISE v AS x, y AS y DRAW point FACET g \
                 LABEL x => 'Value', y => 'Count'"
            )),
            vec![
                (AxisSide::Bottom, "Value".to_string()),
                (AxisSide::Left, "Count".to_string()),
            ]
        );
    }

    #[test]
    fn axis_titles_skip_untitled_axes() {
        // A polar coord has no Cartesian rails to title, and a synthetic dummy
        // position scale has no axis at all.
        assert!(axis_titles(&format!(
            "{FACET_DATA} VISUALISE v AS y, g AS fill DRAW bar PROJECT x, y TO polar"
        ))
        .is_empty());
        assert_eq!(
            axis_titles(&format!("{FACET_DATA} VISUALISE v AS y DRAW bar")),
            vec![(AxisSide::Left, "v".to_string())]
        );
    }

    #[test]
    fn facet_strips_rename_discrete() {
        assert_eq!(
            top_strips(&format!(
                "{FACET_DATA} VISUALISE v AS x, y AS y DRAW point FACET g \
                 SCALE panel RENAMING 'a' => 'Alpha'"
            )),
            vec!["Alpha", "b", "c"]
        );
    }

    #[test]
    fn facet_strips_suppress_discrete() {
        // A suppressed label leaves an empty strip, keeping panel heights aligned.
        assert_eq!(
            top_strips(&format!(
                "{FACET_DATA} VISUALISE v AS x, y AS y DRAW point FACET g \
                 SCALE panel RENAMING 'b' => NULL"
            )),
            vec!["a", "", "c"]
        );
    }

    #[test]
    fn facet_strips_null_level() {
        // A NULL facet level keys as "null" and is renamable under that key.
        let data = "SELECT g, v FROM (VALUES ('a',1),(NULL,2)) t(g,v)";
        assert_eq!(
            top_strips(&format!(
                "{data} VISUALISE v AS x, v AS y DRAW point FACET g \
                 SCALE panel FROM ('a', null)"
            )),
            vec!["a", "null"]
        );
        assert_eq!(
            top_strips(&format!(
                "{data} VISUALISE v AS x, v AS y DRAW point FACET g \
                 SCALE panel FROM ('a', null) RENAMING null => 'The rest'"
            )),
            vec!["a", "The rest"]
        );
    }

    #[test]
    fn facet_strips_binned_ranges() {
        // A numeric facet is binned; strips show the bin range, not the midpoint.
        assert_eq!(
            top_strips(&format!(
                "{FACET_DATA} VISUALISE v AS x, y AS y DRAW point FACET v \
                 SCALE panel SETTING breaks => (0, 10, 20, 30)"
            )),
            vec!["0 – 10", "10 – 20", "20 – 30"]
        );
    }

    #[test]
    fn facet_strips_binned_squish() {
        // `oob => 'squish'` opens the terminal bins: "< upper" / "≥ lower".
        assert_eq!(
            top_strips(&format!(
                "{FACET_DATA} VISUALISE v AS x, y AS y DRAW point FACET v \
                 SCALE panel SETTING breaks => (10, 20, 30), oob => 'squish'"
            )),
            vec!["< 20", "≥ 20"]
        );
    }

    #[test]
    fn facet_strips_binned_closed_right() {
        // `closed => 'right'` flips the open-ended terminal symbols.
        assert_eq!(
            top_strips(&format!(
                "{FACET_DATA} VISUALISE v AS x, y AS y DRAW point FACET v \
                 SCALE panel SETTING breaks => (10, 20, 30), oob => 'squish', \
                 closed => 'right'"
            )),
            vec!["≤ 20", "> 20"]
        );
    }

    #[test]
    fn facet_strips_binned_edge_renaming() {
        // RENAMING applies per break edge, before the range label is built.
        assert_eq!(
            top_strips(&format!(
                "{FACET_DATA} VISUALISE v AS x, y AS y DRAW point FACET v \
                 SCALE panel SETTING breaks => (0, 10, 20, 30) RENAMING 20 => 'twenty'"
            )),
            vec!["0 – 10", "10 – twenty", "twenty – 30"]
        );
    }

    #[test]
    fn facet_strips_binned_reverse() {
        assert_eq!(
            top_strips(&format!(
                "{FACET_DATA} VISUALISE v AS x, y AS y DRAW point FACET v \
                 SCALE panel SETTING breaks => (0, 10, 20, 30), reverse => true"
            )),
            vec!["20 – 30", "10 – 20", "0 – 10"]
        );
    }

    #[test]
    fn facet_strips_binned_temporal() {
        // Temporal binned facets label as date ranges, computed from the typed
        // values rather than from a midpoint string.
        let data = "SELECT CAST(d AS DATE) AS d, v FROM (VALUES \
             ('1973-05-04', 1), ('1973-05-20', 2), ('1973-06-08', 3)) t(d, v)";
        assert_eq!(
            top_strips(&format!(
                "{data} VISUALISE v AS x, v AS y DRAW point FACET d \
                 SCALE panel SETTING breaks => 'month'"
            )),
            vec!["1973-05-01 – 1973-06-01", "1973-06-01 – 1973-07-01"]
        );
    }

    #[test]
    fn facet_strips_null_and_empty_are_separate_panels() {
        // `column_to_strings` renders both a NULL and an empty category as "",
        // so the null flag is what keeps them in separate panels.
        let data = "SELECT g, v FROM (VALUES ('', 1), (NULL, 2), ('a', 3)) t(g, v)";
        assert_eq!(
            top_strips(&format!(
                "{data} VISUALISE v AS x, v AS y DRAW point FACET g"
            )),
            vec!["", "a", "null"]
        );
    }

    #[test]
    fn facet_over_empty_data_is_one_panel() {
        // No levels to lay out: both layouts collapse to the unfaceted single
        // panel rather than building a grid of zero cells.
        let empty = "SELECT g, h, v FROM (VALUES ('a','b',1)) t(g,h,v) WHERE false";
        for query in [
            format!("{empty} VISUALISE v AS x, v AS y DRAW point FACET g"),
            format!("{empty} VISUALISE v AS x, v AS y DRAW point FACET g BY h"),
        ] {
            assert_eq!(strips(&query), vec![(None, None)], "for: {query}");
        }
    }

    #[test]
    fn facet_strips_grid_row_column() {
        // Grid: renamed column labels on the top row only, renamed row labels on
        // the right column only.
        let data = "SELECT r, c, v FROM (VALUES \
             ('r1','c1',1),('r1','c2',2),('r2','c1',3),('r2','c2',4)) t(r,c,v)";
        assert_eq!(
            strips(&format!(
                "{data} VISUALISE v AS x, v AS y DRAW point FACET r BY c \
                 SCALE row RENAMING 'r1' => 'Row one' \
                 SCALE column RENAMING 'c2' => 'Col two'"
            )),
            vec![
                (Some("c1".into()), None),
                (Some("Col two".into()), Some("Row one".into())),
                (None, None),
                (None, Some("r2".into())),
            ]
        );
    }

    #[test]
    fn renders_binned_facet() {
        assert_renders(&format!(
            "{FACET_DATA} VISUALISE v AS x, y AS y DRAW point FACET v \
             SCALE panel SETTING breaks => (0, 10, 20, 30)"
        ));
    }

    #[test]
    fn renders_free_binned_facet() {
        // A free binned position dimension: each panel keeps ggsql's global bin
        // edges but shows only the bins its own data occupies.
        assert_renders(
            "VISUALISE body_mass AS x FROM ggsql:penguins DRAW bar \
             SCALE BINNED x SETTING breaks => (2500, 3500, 4500, 5500, 6500) \
             FACET species SETTING free => 'x'",
        );
    }

    #[test]
    fn renders_binned_size_legend() {
        // A binned *keyed* legend: one key per bin, sized at the bin's midpoint,
        // with ggsql's edge labels on the rail between keys.
        assert_renders(
            "VISUALISE bill_len AS x, bill_dep AS y, body_mass AS size \
             FROM ggsql:penguins DRAW point \
             SCALE BINNED size SETTING breaks => (2500, 3500, 4500, 5500, 6500)",
        );
    }

    #[test]
    fn renders_binned_color_legend() {
        // The same ladder driving color: a stepped colorbar, one block per bin.
        assert_renders(
            "VISUALISE bill_len AS x, bill_dep AS y, body_mass AS color \
             FROM ggsql:penguins DRAW point \
             SCALE BINNED color SETTING breaks => (2500, 3500, 4500, 5500, 6500)",
        );
    }

    #[test]
    fn renders_boxplot_linewidth() {
        // `linewidth` thickens box, whiskers and median alike (VL puts
        // strokeWidth in the boxplot's shared encoding).
        assert_renders(
            "SELECT g, v FROM (VALUES ('a',1),('a',2),('a',3),('a',9),\
             ('b',2),('b',3),('b',4),('b',5)) t(g,v) \
             VISUALISE g AS x, v AS y DRAW boxplot SETTING linewidth => 3",
        );
    }

    #[test]
    fn renders_boxplot_dashed() {
        assert_renders(
            "SELECT g, v FROM (VALUES ('a',1),('a',2),('a',3),('b',2),('b',3),('b',5)) t(g,v) \
             VISUALISE g AS x, v AS y DRAW boxplot \
             SETTING linetype => 'dashed', linewidth => 2",
        );
    }

    #[test]
    fn renders_boxplot_hinge() {
        // `hinge` caps the whiskers with a fixed-size (pt) tick at each fence.
        assert_renders(
            "SELECT g, v FROM (VALUES ('a',1),('a',2),('a',3),('a',9),\
             ('b',2),('b',3),('b',4),('b',5)) t(g,v) \
             VISUALISE g AS x, v AS y DRAW boxplot SETTING hinge => 20",
        );
    }

    #[test]
    fn renders_boxplot_side() {
        // `side` halves the box, median and caps onto one side of the band,
        // leaving whiskers and outliers on the centreline.
        assert_renders(
            "SELECT g, v FROM (VALUES ('a',1),('a',2),('a',3),('a',9),\
             ('b',2),('b',3),('b',4),('b',5)) t(g,v) \
             VISUALISE g AS x, v AS y DRAW boxplot \
             SETTING side => 'right', hinge => 20",
        );
    }

    #[test]
    fn renders_transposed_boxplot() {
        // A horizontal boxplot: ggsql flips the position columns, so the
        // categories are on `pos2` and the summary values in the `pos1` family.
        assert_renders(
            "SELECT g, v FROM (VALUES ('a',1),('a',2),('a',3),('a',9),\
             ('b',2),('b',3),('b',4),('b',5)) t(g,v) \
             VISUALISE v AS x, g AS y DRAW boxplot SETTING hinge => 15",
        );
    }

    #[test]
    fn renders_half_violin_with_half_boxplot() {
        // Opposite `side` values pair the two composites on one band, the
        // documented raincloud-style layout (transposed, so top/bottom).
        assert_renders(
            "SELECT g, v FROM (VALUES ('a',1),('a',2),('a',2),('a',3),('a',4),\
             ('b',2),('b',3),('b',3),('b',4),('b',6)) t(g,v) \
             VISUALISE v AS x, g AS y \
             DRAW violin SETTING side => 'top' \
             DRAW boxplot SETTING side => 'bottom', width => 0.3",
        );
    }

    #[test]
    fn renders_jittered_points() {
        // `position => 'jitter'` spreads the points across their category band;
        // `side` (folded into the offsets by ggsql) keeps them on one half.
        assert_renders(
            "VISUALISE species AS x, bill_len AS y FROM ggsql:penguins DRAW point \
             SETTING position => 'jitter'",
        );
        assert_renders(
            "VISUALISE species AS x, bill_len AS y FROM ggsql:penguins DRAW point \
             SETTING position => 'jitter', side => 'right'",
        );
    }

    #[test]
    fn renders_dodged_points() {
        // Dodge on a geom that doesn't derive its own band edges: the offsets
        // reach the point's band channel.
        assert_renders(
            "SELECT x, g, v FROM (VALUES ('a','p',3),('a','q',5),('b','p',2),('b','q',4)) \
             t(x,g,v) \
             VISUALISE x AS x, v AS y, g AS color DRAW point SETTING position => 'dodge'",
        );
    }

    #[test]
    fn renders_dodged_range_with_hinges() {
        // A dodged interval and its end caps share one offset, so they stay
        // aligned in the dodge slot.
        assert_renders(
            "SELECT g, s, lo, hi FROM (VALUES ('a','p',1,5),('a','q',2,6),('b','p',2,7)) \
             t(g,s,lo,hi) \
             VISUALISE g AS x, lo AS ymin, hi AS ymax, s AS stroke DRAW range \
             SETTING position => 'dodge'",
        );
    }

    #[test]
    fn renders_jitter_with_half_boxplot() {
        // The documented raincloud layout: a one-sided jitter above the
        // centreline, a half-boxplot below it.
        assert_renders(
            "VISUALISE bill_len AS x, species AS y FROM ggsql:penguins \
             DRAW point SETTING position => 'jitter', side => 'top', width => 0.4 \
             DRAW boxplot SETTING side => 'bottom', width => 0.4",
        );
    }

    #[test]
    fn renders_range_hinges() {
        // A range carries 10pt end caps by default; `hinge => null` drops them.
        assert_renders(
            "SELECT g, lo, hi FROM (VALUES ('a',1,5),('b',2,7)) t(g,lo,hi) \
             VISUALISE g AS x, lo AS ymin, hi AS ymax DRAW range",
        );
        assert_renders(
            "SELECT g, lo, hi FROM (VALUES ('a',1,5),('b',2,7)) t(g,lo,hi) \
             VISUALISE g AS y, lo AS xmin, hi AS xmax DRAW range \
             SETTING hinge => 40",
        );
        assert_renders(
            "SELECT g, lo, hi FROM (VALUES ('a',1,5),('b',2,7)) t(g,lo,hi) \
             VISUALISE g AS x, lo AS ymin, hi AS ymax DRAW range \
             SETTING hinge => null",
        );
    }

    #[test]
    fn renders_violin_linewidth() {
        assert_renders(
            "SELECT g, v FROM (VALUES ('a',1),('a',2),('a',2),('a',3),('a',4),\
             ('b',2),('b',3),('b',3),('b',4),('b',6)) t(g,v) \
             VISUALISE g AS x, v AS y DRAW violin \
             SETTING linewidth => 3, linetype => 'dashed'",
        );
    }

    #[test]
    fn renders_dodged_violin() {
        // Two fill groups per category: each must be its own contour (keyed on the
        // category *and* the partition columns), not one merged blob.
        assert_renders(
            "SELECT g, f, v FROM (VALUES ('a','x',1),('a','x',2),('a','x',3),\
             ('a','y',5),('a','y',6),('a','y',7),\
             ('b','x',2),('b','x',3),('b','x',4),('b','y',6),('b','y',7),('b','y',8)) t(g,f,v) \
             VISUALISE g AS x, v AS y, f AS fill DRAW violin",
        );
    }

    #[test]
    fn renders_text_stroke() {
        // A constant `stroke` outlines the glyphs; white-on-dark legibility.
        assert_renders(
            "SELECT 1 AS x, 2 AS y, 'peak' AS lbl UNION ALL SELECT 2, 3, 'trough' \
             VISUALISE x AS x, y AS y, lbl AS label DRAW text \
             SETTING fontsize => 28, fontweight => 'bold', color => 'black', \
             stroke => 'white'",
        );
    }

    #[test]
    fn renders_text_stroke_by_group() {
        // A data-mapped outline color: one scale + legend, per-row outline.
        assert_renders(
            "SELECT 1 AS x, 2 AS y, 'a' AS lbl, 'one' AS g \
             UNION ALL SELECT 2, 3, 'b', 'two' \
             VISUALISE x AS x, y AS y, lbl AS label, g AS stroke DRAW text \
             SETTING fontsize => 30, fontweight => 'bold'",
        );
    }

    #[test]
    fn renders_titled_plot() {
        // Title, subtitle and caption all sit on the composition, above/below the
        // single panel.
        assert_renders(
            "SELECT 1 AS x, 2 AS y UNION ALL SELECT 2, 3 UNION ALL SELECT 3, 1 \
             VISUALISE x AS x, y AS y DRAW point \
             LABEL title => 'Sales by Region', subtitle => 'FY 2024', \
             caption => 'Source: internal'",
        );
    }

    #[test]
    fn renders_suppressed_title() {
        // `LABEL title => NULL` suppresses; the subtitle still renders.
        assert_renders(
            "SELECT 1 AS x, 2 AS y UNION ALL SELECT 2, 3 \
             VISUALISE x AS x, y AS y DRAW point \
             LABEL title => NULL, subtitle => 'no title above me'",
        );
    }

    #[test]
    fn renders_titled_facet() {
        // One composition-spanning title over the whole 3-panel strip, not one
        // title per panel.
        assert_renders(
            "SELECT x, y, g FROM (VALUES (1,1,'a'),(2,2,'a'),(1,2,'b'),(2,3,'b'),\
             (1,3,'c'),(2,1,'c')) t(x,y,g) \
             VISUALISE x AS x, y AS y DRAW point FACET g \
             LABEL title => 'One title for all panels'",
        );
    }

    #[test]
    fn map_range_pads_like_vegalite() {
        // 10% of the span, split evenly around the centre — the same framing
        // Vega-Lite's projection fit produces from `span * 1.1`.
        let r = compose::map_range(0.0, 10.0);
        assert_eq!(*r.start(), -0.5);
        assert_eq!(*r.end(), 10.5);
        assert_eq!((r.end() - r.start()) / 10.0, 1.1);
    }

    #[test]
    fn map_range_widens_a_degenerate_extent() {
        // A single point has no span to pad, so it is widened to a mappable one.
        let r = compose::map_range(3.0, 3.0);
        assert_eq!(*r.start(), 2.5);
        assert_eq!(*r.end(), 3.5);
    }

    #[test]
    fn map_range_orients_an_inverted_extent() {
        // A reversed bbox reaches here — `map_bbox` checks its four numbers
        // for finiteness, not for order — and a scale cannot map an inverted
        // range at all, so it is oriented rather than passed through.
        let r = compose::map_range(10.0, 0.0);
        assert_eq!(*r.start(), -0.5);
        assert_eq!(*r.end(), 10.5);
    }

    /// `arrow` is a stub in both writers, so the rejection lives in the shared
    /// composition layer rather than in any one of them.
    #[test]
    fn rejects_unsupported_geom() {
        let spec = spec_for(
            "SELECT 0 AS x, 0 AS y, 1 AS xend, 1 AS yend \
             VISUALISE x AS x, y AS y, xend AS xend, yend AS yend DRAW arrow",
        );
        let err = compose::validate_plot(spec.plot()).unwrap_err();
        assert!(matches!(err, GgsqlError::WriterError(_)));
        assert!(err.to_string().contains("'arrow' geom"), "{err}");
    }
}

// SVG output is readable text, so these check directly that the breaks, labels
// and titles ggsql resolved are the ones reaching the output — which a PNG
// cannot show. None of it needs a GPU.
#[cfg(all(test, feature = "duckdb", feature = "svg"))]
mod svg_text {
    use super::*;
    use crate::reader::{DuckDBReader, Reader};
    use crate::writer::{Writer, WriterOptions};

    const FACET_DATA: &str = "SELECT g, v, y FROM (VALUES \
         ('a',5,1),('a',7,2),('b',15,3),('b',18,1),('c',25,2),('c',28,3)) t(g,v,y)";

    fn svg(query: &str) -> String {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let spec = reader.execute(query).unwrap();
        let (svg, warnings) = SvgWriter::new(640, 480, 96.0)
            .render_reporting(&spec)
            .unwrap_or_else(|e| panic!("svg render failed: {e}"));
        assert!(warnings.is_empty(), "svg degraded the plot: {warnings:?}");
        svg
    }

    /// Every `<text>` element's text, in document order.
    ///
    /// A run of styled spans is one string, so a markdown-emphasised title
    /// reads as the sentence a user typed rather than as its pieces.
    fn texts(svg: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut rest = svg;
        while let Some(open) = rest.find("<text") {
            rest = &rest[open..];
            let Some(end) = rest.find("</text>") else {
                break;
            };
            let element = &rest[..end];
            let mut label = String::new();
            let mut span = element;
            while let Some(at) = span.find("<tspan") {
                span = &span[at..];
                let Some(gt) = span.find('>') else { break };
                let Some(close) = span.find("</tspan>") else {
                    break;
                };
                label.push_str(&span[gt + 1..close]);
                span = &span[close..];
            }
            out.push(label);
            rest = &rest[end..];
        }
        out
    }

    fn contains(svg: &str, label: &str) -> bool {
        texts(svg).iter().any(|t| t == label)
    }

    #[test]
    fn tick_labels_are_the_ones_ggsql_resolved() {
        // ggsql's own break spacing and number formatting: the trailing `.0`
        // shows these are pass-throughs, not the renderer's idea of a nice tick.
        let linear = svg("SELECT x, y FROM (VALUES (1,2),(2,3),(3,1)) t(x,y) \
             VISUALISE x AS x, y AS y DRAW point");
        for label in ["1.0", "1.5", "2.0", "2.5", "3.0"] {
            assert!(contains(&linear, label), "missing tick '{label}'");
        }

        // A `RENAMING` on a discrete axis reaches the rail, and the axis does
        // not shift: the break is kept and only its label replaced.
        let renamed = svg("SELECT c, v FROM (VALUES ('a',3),('b',5)) t(c,v) \
             VISUALISE c AS x, v AS y DRAW bar SCALE x RENAMING 'a' => 'Alpha'");
        assert!(contains(&renamed, "Alpha"));
        assert!(contains(&renamed, "b"));
    }

    /// A log axis should carry decade ticks, but `Scale::numeric_breaks()`
    /// resolves `[5e-308, 2e-256, …, 100]` for a 1–100 log10 domain and both
    /// writers pass that through. Kept as a failing expectation so it turns
    /// green on its own once the scale is fixed.
    #[test]
    #[ignore = "ggsql resolves log-scale breaks to denormals; not a writer bug"]
    fn log_tick_labels_should_be_decades() {
        let log = svg("SELECT x, y FROM (VALUES (1,2),(10,3),(100,1)) t(x,y) \
             VISUALISE x AS x, y AS y DRAW point SCALE x VIA log");
        for label in ["1", "10", "100"] {
            assert!(contains(&log, label), "missing log tick '{label}'");
        }
    }

    #[test]
    fn a_binned_scales_edge_labels_reach_the_legend() {
        // The renderer cannot derive a bin ladder, so all five edges appearing
        // verbatim on the colorbar rail is the pass-through.
        let binned = svg(
            "VISUALISE bill_len AS x, bill_dep AS y, body_mass AS color \
             FROM ggsql:penguins DRAW point \
             SCALE BINNED color SETTING breaks => (2500, 3500, 4500, 5500, 6500)",
        );
        for label in ["2500", "3500", "4500", "5500", "6500"] {
            assert!(contains(&binned, label), "missing bin edge '{label}'");
        }
    }

    #[test]
    fn facet_strip_labels_appear_once_each_in_panel_order() {
        let faceted = svg(&format!(
            "{FACET_DATA} VISUALISE v AS x, y AS y DRAW point FACET g"
        ));
        let labels = texts(&faceted);
        for level in ["a", "b", "c"] {
            assert_eq!(
                labels.iter().filter(|t| *t == level).count(),
                1,
                "strip '{level}' should appear exactly once in {labels:?}"
            );
        }
        // In panel order, which is the facet scale's resolved order.
        let order: Vec<&String> = labels
            .iter()
            .filter(|t| ["a", "b", "c"].contains(&t.as_str()))
            .collect();
        assert_eq!(order, vec!["a", "b", "c"]);

        // And `RENAMING` reaches the strip, since the label is ggsql's.
        let renamed = svg(&format!(
            "{FACET_DATA} VISUALISE v AS x, y AS y DRAW point FACET g \
             SCALE panel RENAMING 'a' => 'Alpha'"
        ));
        assert!(contains(&renamed, "Alpha"));
        assert!(!contains(&renamed, "a"));
    }

    #[test]
    fn binned_facet_strips_show_ggsqls_range_labels() {
        let binned = svg(&format!(
            "{FACET_DATA} VISUALISE v AS x, y AS y DRAW point FACET v \
             SCALE panel SETTING breaks => (0, 10, 20, 30)"
        ));
        for label in ["0 – 10", "10 – 20", "20 – 30"] {
            assert!(contains(&binned, label), "missing strip '{label}'");
        }
    }

    #[test]
    fn every_plot_label_reaches_the_output() {
        let labelled = svg("SELECT x, y FROM (VALUES (1,2),(2,3)) t(x,y) \
             VISUALISE x AS x, y AS y DRAW point \
             LABEL title => 'The title', subtitle => 'The subtitle', \
             caption => 'The caption', x => 'Across', y => 'Up'");
        for label in ["The title", "The subtitle", "The caption", "Across", "Up"] {
            assert!(contains(&labelled, label), "missing label '{label}'");
        }
    }

    #[test]
    fn markdown_in_a_label_is_parsed_rather_than_printed() {
        let emphasised = svg("SELECT x, y FROM (VALUES (1,2),(2,3)) t(x,y) \
             VISUALISE x AS x, y AS y DRAW point LABEL title => 'A *bold* title'");
        // The words survive, the markers do not.
        assert!(contains(&emphasised, "A bold title"));
        assert!(
            !texts(&emphasised).iter().any(|t| t.contains('*')),
            "a literal '*' means the markdown was not parsed"
        );
        // And the emphasised run is styled, not merely re-joined.
        assert!(
            emphasised.contains("font-style=\"italic\""),
            "the emphasised span carries no style"
        );
    }

    #[test]
    fn a_legends_title_and_key_labels_appear() {
        // The title is the mapped column, and the keys are the categorical
        // domain ggsql trained, in its resolved order.
        let keyed = svg(
            "SELECT x, y, g FROM (VALUES (1,2,'alpha'),(2,3,'beta'),(3,1,'alpha')) t(x,y,g) \
             VISUALISE x AS x, y AS y, g AS color DRAW point",
        );
        for label in ["g", "alpha", "beta"] {
            assert!(contains(&keyed, label), "missing legend text '{label}'");
        }

        // `RENAMING` relabels a key without dropping the others.
        let renamed = svg(
            "SELECT x, y, g FROM (VALUES (1,2,'alpha'),(2,3,'beta'),(3,1,'alpha')) t(x,y,g) \
             VISUALISE x AS x, y AS y, g AS color DRAW point \
             SCALE color RENAMING 'alpha' => 'First'",
        );
        assert!(contains(&renamed, "First"));
        assert!(contains(&renamed, "beta"));
        assert!(!contains(&renamed, "alpha"));
    }

    #[test]
    fn outline_mode_turns_text_into_paths() {
        let query = "SELECT x, y FROM (VALUES (1,2),(2,3)) t(x,y) \
                     VISUALISE x AS x, y AS y DRAW point LABEL title => 'Outlined'";
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let spec = reader.execute(query).unwrap();

        let as_text = SvgWriter::new(640, 480, 96.0).render(&spec).unwrap();
        let as_paths = SvgWriter::new(640, 480, 96.0)
            .outline_text(true)
            .render(&spec)
            .unwrap();

        assert!(as_text.contains("<text"), "text mode should emit <text>");
        assert!(
            !as_paths.contains("<text"),
            "outline mode should emit no <text>"
        );
        assert!(
            as_paths.matches("<path").count() > as_text.matches("<path").count(),
            "outlined glyphs should add paths"
        );
    }

    #[test]
    fn units_decide_what_the_root_element_declares() {
        let query = "SELECT x, y FROM (VALUES (1,2),(2,3)) t(x,y) \
                     VISUALISE x AS x, y AS y DRAW point";
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let spec = reader.execute(query).unwrap();

        let build = |pairs: &[&str]| {
            let options = WriterOptions::parse(pairs).unwrap();
            SvgWriter::from_options(&options)
                .unwrap()
                .render(&spec)
                .unwrap()
        };

        // Six inches at 300 dpi: an 1800 px viewBox on a 432 pt root, so the
        // file prints six inches wide.
        let physical = build(&["width=6", "height=4", "units=in", "dpi=300"]);
        assert!(
            physical.contains("width=\"432pt\""),
            "root: {}",
            &physical[..120]
        );
        assert!(physical.contains("viewBox=\"0 0 1800 1200\""));

        // A pixel canvas stays in pixels.
        let pixels = build(&["width=1800", "height=1200", "dpi=300"]);
        assert!(
            pixels.contains("width=\"1800\""),
            "root: {}",
            &pixels[..120]
        );
        assert!(pixels.contains("viewBox=\"0 0 1800 1200\""));
    }

    #[test]
    fn an_id_prefix_namespaces_generated_ids() {
        let query = "SELECT x, y, c FROM (VALUES (1,2,10),(2,3,50),(3,1,90)) t(x,y,c) \
                     VISUALISE x AS x, y AS y, c AS color DRAW point";
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let spec = reader.execute(query).unwrap();

        let bare = SvgWriter::new(640, 480, 96.0).render(&spec).unwrap();
        let prefixed = SvgWriter::new(640, 480, 96.0)
            .id_prefix("fig1-")
            .render(&spec)
            .unwrap();

        assert!(bare.contains("id=\"c0\""), "expected an unprefixed id");
        assert!(!prefixed.contains("id=\"c0\""));
        assert!(prefixed.contains("id=\"fig1-c0\""));
        // Every reference is rewritten too, or the file is broken rather than
        // merely renamed.
        assert!(!prefixed.contains("url(#c0)"));
        assert!(prefixed.contains("url(#fig1-c0)"));
    }
}

// PDF is not readable text, but its structure is checkable — and, like SVG,
// without a GPU.
#[cfg(all(test, feature = "duckdb", feature = "pdf"))]
mod pdf_structure {
    use super::*;
    use crate::reader::{DuckDBReader, Reader};
    use crate::writer::Writer;

    fn spec() -> crate::reader::Spec {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute(
                "SELECT x, y FROM (VALUES (1,2),(2,3),(3,1)) t(x,y) \
                 VISUALISE x AS x, y AS y DRAW point LABEL title => 'A page'",
            )
            .unwrap()
    }

    #[test]
    fn the_page_box_is_the_canvas_at_seventy_two_points_per_inch() {
        // 640 px at 96 dpi is 6⅔ in, which is 480 pt; 480 px is 360 pt.
        let pdf = PdfWriter::new(640, 480, 96.0)
            .compress(false)
            .render(&spec())
            .unwrap();
        let text = String::from_utf8_lossy(&pdf);
        assert!(
            text.contains("/MediaBox [0 0 480 360]"),
            "unexpected page box"
        );
    }

    #[test]
    fn an_uncompressed_page_is_readable_and_a_compressed_one_is_smaller() {
        let readable = PdfWriter::new(640, 480, 96.0)
            .compress(false)
            .render(&spec())
            .unwrap();
        let compressed = PdfWriter::new(640, 480, 96.0).render(&spec()).unwrap();

        assert!(readable.starts_with(b"%PDF-"));
        assert!(compressed.starts_with(b"%PDF-"));
        // `compress=false` exists so a user can read or diff the stream.
        assert!(!String::from_utf8_lossy(&readable).contains("/FlateDecode"));
        assert!(String::from_utf8_lossy(&compressed).contains("/FlateDecode"));
        assert!(compressed.len() < readable.len());
    }

    #[test]
    fn a_font_is_subset_into_the_page() {
        // The page must carry its own glyphs, or a figure in a paper renders
        // in whatever the reader substitutes.
        let pdf = PdfWriter::new(640, 480, 96.0)
            .compress(false)
            .render(&spec())
            .unwrap();
        let text = String::from_utf8_lossy(&pdf);
        assert!(text.contains("/FontFile2"), "no embedded font programme");
        assert!(text.contains("/Type /Font"));
    }
}

// The `hep` round trip: a document is written from a live composition, read
// back into a new one, and both are rendered to SVG and compared byte for byte.
// Any scale, break, label, theme entry, channel or geom that fails to survive
// shows up as different drawing commands. SVG rather than raster because it is
// deterministic; GPU antialiasing is not. Test-only — the library only writes.
#[cfg(all(
    test,
    feature = "duckdb",
    feature = "hep-read",
    feature = "svg",
    feature = "builtin-data"
))]
mod hep_roundtrip {
    use super::*;
    use crate::reader::{DuckDBReader, Reader};
    // Driven directly rather than through the writer, so a loss shows up as a
    // drawing difference rather than as writer configuration.
    use hephaestus::document::{
        read_composition, read_hints, unsupported_items_for, write_composition, ReadContext,
        WriteOptions,
    };

    /// Deliberately broad: several geoms, a facet, a legend, markdown chrome,
    /// and a transform — so the round trip is not proved on a scatter plot.
    const QUERIES: &[(&str, &str)] = &[
        (
            "faceted scatter with a legend",
            "VISUALISE bill_len AS x, bill_dep AS y, species AS color \
             FROM ggsql:penguins DRAW point FACET island \
             LABEL title => 'Penguin *bills*', caption => 'From ggsql:penguins'",
        ),
        (
            "multi-layer with a colorbar",
            "VISUALISE bill_len AS x, bill_dep AS y, body_mass AS color \
             FROM ggsql:penguins DRAW point DRAW line",
        ),
        (
            "boxplot with a free facet",
            "VISUALISE species AS x, body_mass AS y, species AS fill \
             FROM ggsql:penguins DRAW boxplot FACET island \
             SETTING free => ('y')",
        ),
    ];

    fn compose_for(query: &str) -> hephaestus::plot::PlotComposition {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let spec = reader.execute(query).unwrap();
        compose::validate_plot(spec.plot()).unwrap();
        compose::build_composition(spec.plot(), spec.data()).unwrap()
    }

    fn to_svg(view: &mut hephaestus::plot::PlotComposition) -> String {
        use hephaestus::geometry::Size;
        use hephaestus::svg::{encode_svg, SvgScene};
        let size = Size::new(640.0, 480.0);
        let mut scene = SvgScene::new(size, 96.0);
        view.render(&mut scene, size, 96.0);
        assert!(
            scene.warnings().is_empty(),
            "svg degraded the plot: {:?}",
            scene.warnings()
        );
        encode_svg(&scene)
    }

    #[test]
    fn a_document_rebuilds_the_plot_it_captured() {
        for (label, query) in QUERIES {
            let mut live = compose_for(query);
            let bytes = write_composition(&live, &WriteOptions::default())
                .unwrap_or_else(|e| panic!("{label}: write failed: {e}"));
            let mut rebuilt = read_composition(&bytes, ReadContext::builtin())
                .unwrap_or_else(|e| panic!("{label}: read failed: {e}"));

            assert_eq!(
                to_svg(&mut rebuilt),
                to_svg(&mut live),
                "{label}: the rebuilt composition draws differently"
            );
        }
    }

    /// The projection has to be restored before the axes are attached, or a
    /// polar placement is validated against the Cartesian default and rejected.
    #[test]
    fn a_polar_document_rebuilds_too() {
        let query = "SELECT c FROM (VALUES ('a'),('a'),('a'),('b'),('b'),('c')) t(c) \
                     VISUALISE c AS fill DRAW bar PROJECT TO polar";
        let mut live = compose_for(query);
        let bytes = write_composition(&live, &WriteOptions::default()).unwrap();
        let mut rebuilt = read_composition(&bytes, ReadContext::builtin()).unwrap();
        assert_eq!(to_svg(&mut rebuilt), to_svg(&mut live));
    }

    #[test]
    fn nothing_ggsql_draws_is_beyond_the_format() {
        // A failure means the writer grew something the format cannot name.
        for (label, query) in QUERIES {
            let view = compose_for(query);
            let problems = unsupported_items_for(&view, &WriteOptions::default());
            assert!(problems.is_empty(), "{label}: {problems:?}");
        }
    }

    #[test]
    fn the_writers_hints_travel_with_the_document() {
        let spec = {
            let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
            reader.execute(QUERIES[0].1).unwrap()
        };
        let (bytes, warnings) = HepWriter::new(1600, 900, 150.0)
            .background(rgba(0.0, 0.0, 0.0, 1.0))
            .render_reporting(&spec)
            .unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        assert!(bytes.starts_with(b"HEPHPLOT"), "missing the format's magic");

        let hints = read_hints(&bytes).unwrap();
        assert_eq!(hints.size, Some((1600.0, 900.0)));
        assert_eq!(hints.dpi, Some(150.0));
        assert_eq!(
            hints.background.map(|c| c.components),
            Some([0.0, 0.0, 0.0, 1.0])
        );

        // A transparent background is a request, not the absence of one: read
        // back as `None` a consumer paints white behind the plot instead.
        let (transparent, _) = HepWriter::new(1600, 900, 150.0)
            .background(rgba(0.0, 0.0, 0.0, 0.0))
            .render_reporting(&spec)
            .unwrap();
        let hints = read_hints(&transparent).unwrap();
        assert_eq!(
            hints.background.map(|c| c.components),
            Some([0.0, 0.0, 0.0, 0.0])
        );
    }

    #[test]
    fn a_document_survives_a_render_at_a_different_size() {
        // The whole point of the format: the consumer picks the size, and the
        // composition re-solves its layout for it.
        use hephaestus::geometry::Size;
        use hephaestus::svg::{encode_svg, SvgScene};

        let live = compose_for(QUERIES[0].1);
        let bytes = write_composition(&live, &WriteOptions::default()).unwrap();
        let mut rebuilt = read_composition(&bytes, ReadContext::builtin()).unwrap();

        for size in [Size::new(320.0, 240.0), Size::new(1600.0, 900.0)] {
            let mut scene = SvgScene::new(size, 96.0);
            rebuilt.render(&mut scene, size, 96.0);
            let svg = encode_svg(&scene);
            assert!(svg.contains(&format!("viewBox=\"0 0 {} {}\"", size.width, size.height)));
            assert!(svg.matches("<path").count() > 0);
        }
    }
}
