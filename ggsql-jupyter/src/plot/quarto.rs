//! Reading Quarto's figure settings out of the environment.
//!
//! Quarto's jupyter engine sets four environment variables on the kernel
//! process (`quarto/share/jupyter/notebook.py`) saying exactly what figure it
//! wants. They map one-to-one onto what the writers take, so a PDF document
//! gets a vector figure with embedded fonts, an HTML one a correctly-sized
//! raster at the requested dpi, and `fig-width: 6` means six inches.

use super::sizing::Canvas;
use super::Format;

/// What Quarto asked for, when it asked.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuartoFigure {
    pub format: Format,
    pub canvas: Canvas,
}

/// Read Quarto's figure settings from the process environment.
///
/// `None` when `QUARTO_FIG_FORMAT` is unset or names something we cannot
/// produce — which includes the plain-Jupyter case, where nothing is set at
/// all. The caller falls back to its own default.
pub fn from_env() -> Option<QuartoFigure> {
    from_vars(|key| std::env::var(key).ok())
}

/// [`from_env`] against an arbitrary source, so the mapping is testable without
/// touching the global process environment.
pub fn from_vars(get: impl Fn(&str) -> Option<String>) -> Option<QuartoFigure> {
    let format = match get("QUARTO_FIG_FORMAT")?.trim().to_lowercase().as_str() {
        "png" => Format::Png,
        "jpeg" | "jpg" => Format::Jpeg,
        "svg" => Format::Svg,
        "pdf" => Format::Pdf,
        // Quarto normalises `retina` to `png` at doubled dpi before it reaches
        // us; seeing it here means a version that does not, and png still fits.
        "retina" => Format::Png,
        _ => return None,
    };

    let number = |key: &str| get(key).and_then(|v| v.trim().parse::<f64>().ok());
    // Quarto's width and height are inches; its dpi is what to render them at.
    let width = number("QUARTO_FIG_WIDTH").filter(|v| *v > 0.0);
    let height = number("QUARTO_FIG_HEIGHT").filter(|v| *v > 0.0);
    let dpi = number("QUARTO_FIG_DPI").filter(|v| *v > 0.0);

    // The only one of the three with a real default, resolved once here so
    // `fig-dpi: 300` works with or without a size alongside it.
    let dpi = dpi.unwrap_or(super::sizing::CSS_DPI);
    let canvas = match (width, height) {
        (Some(w), Some(h)) => Canvas::from_inches(w, h, dpi),
        // A document that sets one dimension and not the other is unusual, but
        // the golden ratio is a better guess than refusing the figure.
        (Some(w), None) => Canvas::from_inches(w, w / 1.618, dpi),
        (None, Some(h)) => Canvas::from_inches(h * 1.618, h, dpi),
        (None, None) => Canvas::default_at(dpi),
    };

    Some(QuartoFigure { format, canvas })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn vars(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        move |key: &str| map.get(key).cloned()
    }

    #[test]
    fn nothing_set_is_not_a_quarto_render() {
        // The plain-Jupyter case: no variables at all.
        assert!(from_vars(vars(&[])).is_none());
    }

    #[test]
    fn each_format_maps_to_its_writer() {
        let cases = [
            ("png", Format::Png),
            ("jpeg", Format::Jpeg),
            ("jpg", Format::Jpeg),
            ("svg", Format::Svg),
            ("pdf", Format::Pdf),
            // Quarto doubles the dpi itself, so this must not be doubled again.
            ("retina", Format::Png),
        ];
        for (value, expected) in cases {
            let figure = from_vars(vars(&[("QUARTO_FIG_FORMAT", value)])).unwrap();
            assert_eq!(figure.format, expected, "{value}");
        }
    }

    #[test]
    fn a_format_we_cannot_produce_is_declined_rather_than_guessed() {
        assert!(from_vars(vars(&[("QUARTO_FIG_FORMAT", "gif")])).is_none());
        assert!(from_vars(vars(&[("QUARTO_FIG_FORMAT", "")])).is_none());
    }

    #[test]
    fn inches_and_dpi_become_a_canvas() {
        let figure = from_vars(vars(&[
            ("QUARTO_FIG_FORMAT", "png"),
            ("QUARTO_FIG_WIDTH", "6"),
            ("QUARTO_FIG_HEIGHT", "4"),
            ("QUARTO_FIG_DPI", "150"),
        ]))
        .unwrap();
        // `fig-width: 6` means six inches, so 900 px at 150 dpi.
        assert_eq!(figure.canvas.width, 900);
        assert_eq!(figure.canvas.height, 600);
        assert_eq!(figure.canvas.dpi, 150.0);
    }

    #[test]
    fn a_format_without_a_size_still_renders() {
        let figure = from_vars(vars(&[("QUARTO_FIG_FORMAT", "pdf")])).unwrap();
        assert_eq!(figure.format, Format::Pdf);
        assert_eq!(figure.canvas, Canvas::default());
    }

    #[test]
    fn a_dpi_without_a_size_still_sets_the_resolution() {
        // `fig-dpi: 300` alone: the default figure at print resolution rather
        // than a silent 96 dpi.
        let figure = from_vars(vars(&[
            ("QUARTO_FIG_FORMAT", "png"),
            ("QUARTO_FIG_DPI", "300"),
        ]))
        .unwrap();
        assert_eq!(figure.canvas.dpi, 300.0);
        assert_eq!(figure.canvas.css_size(), Canvas::default().css_size());
    }

    #[test]
    fn unusable_numbers_fall_back_rather_than_failing_the_render() {
        let figure = from_vars(vars(&[
            ("QUARTO_FIG_FORMAT", "png"),
            ("QUARTO_FIG_WIDTH", "not-a-number"),
            ("QUARTO_FIG_DPI", "0"),
        ]))
        .unwrap();
        assert_eq!(figure.canvas, Canvas::default());
    }
}
